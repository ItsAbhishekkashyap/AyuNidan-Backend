import { AsyncLocalStorage } from 'node:async_hooks';

interface RequestContext {
  requestId: string;
}

const storage = new AsyncLocalStorage<RequestContext>();

export const runWithRequestContext = <T>(context: RequestContext, fn: () => T): T => storage.run(context, fn);

export const getRequestId = (): string | undefined => storage.getStore()?.requestId;
