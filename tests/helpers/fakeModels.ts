/**
 * In-memory stand-ins for the Mongoose models so route tests run without MongoDB.
 * They implement only the query shapes the controllers/services use.
 */
import { Types } from 'mongoose';

type Doc = Record<string, any>;

export const db = {
  users: new Map<string, Doc>(),
  consultations: [] as Doc[],
  documents: [] as Doc[],
  references: [] as Doc[],
};

/** When set, the next write on the named model throws (simulates DB failures). */
export const faults = { failNextConsultationUpdate: false, failConsultationCreate: false };

export const resetDb = (): void => {
  db.users.clear();
  db.consultations.length = 0;
  db.documents.length = 0;
  db.references.length = 0;
  faults.failNextConsultationUpdate = false;
  faults.failConsultationCreate = false;
};

const query = <T>(produce: () => T) => {
  const q: any = {};
  for (const method of ['sort', 'skip', 'limit', 'lean', 'select']) q[method] = () => q;
  q.exec = async () => produce();
  q.then = (resolve: any, reject: any) => Promise.resolve().then(produce).then(resolve, reject);
  return q;
};

const getPath = (doc: Doc, key: string): unknown => key.split('.').reduce<any>((v, k) => v?.[k], doc);

const matchValue = (actual: unknown, expected: any): boolean => {
  if (expected && typeof expected === 'object' && !Array.isArray(expected) && !(expected instanceof Types.ObjectId)) {
    if ('$regex' in expected) return new RegExp(expected.$regex, expected.$options).test(String(actual ?? ''));
    if ('$in' in expected) return expected.$in.map(String).includes(String(actual));
    if ('$nin' in expected) return !expected.$nin.map(String).includes(String(actual));
    if ('$ne' in expected) return String(actual) !== String(expected.$ne);
  }
  return String(actual) === String(expected);
};

export const matches = (doc: Doc, filter: Doc): boolean =>
  Object.entries(filter).every(([key, expected]) => matchValue(getPath(doc, key), expected));

const applyUpdate = (doc: Doc, update: Doc): void => {
  for (const [key, value] of Object.entries(update.$set ?? {})) doc[key] = value;
  for (const key of Object.keys(update.$unset ?? {})) delete doc[key];
  if (!update.$set && !update.$unset) Object.assign(doc, update);
};

const makeCollection = (store: () => Doc[], hooks: { beforeCreate?: () => void; beforeUpdate?: () => void } = {}) => ({
  find: (filter: Doc) => query(() => store().filter((d) => matches(d, filter)).map((d) => ({ ...d }))),
  findOne: (filter: Doc) => query(() => {
    const found = store().find((d) => matches(d, filter));
    return found ? { ...found } : null;
  }),
  countDocuments: (filter: Doc) => query(() => store().filter((d) => matches(d, filter)).length),
  findOneAndDelete: (filter: Doc) =>
    query(() => {
      const index = store().findIndex((d) => matches(d, filter));
      return index < 0 ? null : store().splice(index, 1)[0];
    }),
  findOneAndUpdate: (filter: Doc, update: Doc) =>
    query(() => {
      hooks.beforeUpdate?.();
      const found = store().find((d) => matches(d, filter));
      if (!found) return null;
      applyUpdate(found, update);
      return { ...found };
    }),
  updateMany: (filter: Doc, update: Doc) =>
    query(() => {
      const found = store().filter((d) => matches(d, filter));
      found.forEach((d) => applyUpdate(d, update));
      return { modifiedCount: found.length };
    }),
  deleteMany: (filter: Doc) =>
    query(() => {
      const before = store().length;
      const keep = store().filter((d) => !matches(d, filter));
      store().length = 0;
      store().push(...keep);
      return { deletedCount: before - keep.length };
    }),
  create: async (doc: Doc) => {
    hooks.beforeCreate?.();
    const created = {
      _id: new Types.ObjectId().toString(),
      createdAt: new Date().toISOString(),
      ...doc,
      ...(doc.userId ? { userId: String(doc.userId) } : {}),
    };
    store().push(created);
    return { ...created };
  },
});

export const FakeConsultation = {
  ...makeCollection(() => db.consultations, {
    beforeCreate: () => {
      if (faults.failConsultationCreate) throw new Error('simulated create failure');
    },
    beforeUpdate: () => {
      if (faults.failNextConsultationUpdate) {
        faults.failNextConsultationUpdate = false;
        throw new Error('simulated update failure');
      }
    },
  }),
  aggregate: async (pipeline: Doc[]) => {
    const match = pipeline[0].$match;
    const groups = new Map<string, { count: number; total: number }>();
    for (const doc of db.consultations.filter((d) => matches(d, match))) {
      const group = groups.get(doc.riskLevel) ?? { count: 0, total: 0 };
      group.count += 1;
      group.total += doc.riskScore ?? 0;
      groups.set(doc.riskLevel, group);
    }
    return [...groups].map(([riskLevel, g]) => ({ riskLevel, count: g.count, averageScore: g.total / g.count }));
  },
};

export const FakeDocument = makeCollection(() => db.documents);

export const FakeReferenceSource = makeCollection(() => db.references);

export const FakeUser = {
  findById: (id: string) => query(() => (db.users.has(String(id)) ? { _id: id } : null)),
  findOne: (filter: Doc) => query(() => [...db.users.values()].find((u) => u.email === filter.email) ?? null),
  create: async (doc: Doc) => {
    const created = { _id: new Types.ObjectId(), ...doc, save: async () => undefined };
    db.users.set(created._id.toString(), created);
    return created;
  },
};
