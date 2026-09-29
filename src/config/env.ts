import { z } from 'zod';

/**
 * Validated application configuration. Required settings have NO fallbacks:
 * the server refuses to start if they are missing or weak.
 * Error messages name variables only — never their values.
 */
const EnvSchema = z.object({
  NODE_ENV: z.string().default('development'),
  PORT: z.coerce.number().int().positive().default(8080),
  MONGODB_URI: z.string().trim().min(1, 'MONGODB_URI is required'),
  JWT_SECRET: z
    .string({ error: 'JWT_SECRET is required' })
    .min(32, 'JWT_SECRET must be at least 32 characters'),
  JWT_EXPIRES_IN: z
    .string()
    .regex(/^\d+[smhd]$/, 'JWT_EXPIRES_IN must look like 15m, 12h or 7d')
    .default('7d'),
  GOOGLE_CLIENT_ID: z.string().trim().optional(),
  /** Express "trust proxy" setting: e.g. 1 (one proxy hop), true, false, "loopback". */
  TRUST_PROXY: z.string().trim().optional(),
});

export type AppConfig = z.infer<typeof EnvSchema>;

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigError';
  }
}

let cached: AppConfig | null = null;

export const loadConfig = (env: NodeJS.ProcessEnv = process.env): AppConfig => {
  const result = EnvSchema.safeParse(env);
  if (!result.success) {
    const problems = result.error.issues.map((issue) => `- ${issue.path.join('.')}: ${issue.message}`);
    throw new ConfigError(`Invalid configuration:\n${problems.join('\n')}`);
  }
  return result.data;
};

/** Memoised validated config. Throws ConfigError if required settings are missing. */
export const getConfig = (): AppConfig => {
  if (!cached) cached = loadConfig();
  return cached;
};

/** Test helper. */
export const resetConfigCache = (): void => {
  cached = null;
};

export const parseTrustProxy = (value: string | undefined): boolean | number | string => {
  if (value === undefined || value === '') return false;
  if (value === 'true') return true;
  if (value === 'false') return false;
  const asNumber = Number(value);
  return Number.isInteger(asNumber) ? asNumber : value;
};
