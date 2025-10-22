export type ConfigValue = string | number | boolean | null | ConfigValue[] | ConfigMap;
export type ConfigMap = { [key: string]: ConfigValue };

export const isObject = (value: ConfigValue): value is ConfigMap =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

export const updateAtPath = (value: ConfigValue, path: (string | number)[], next: ConfigValue): ConfigValue => {
  if (path.length === 0) {
    return next;
  }
  const [head, ...rest] = path;
  if (Array.isArray(value)) {
    const index = typeof head === 'number' ? head : parseInt(head, 10);
    const clone = [...value];
    clone[index] = updateAtPath(clone[index], rest, next);
    return clone;
  }
  if (isObject(value)) {
    const key = String(head);
    const clone: ConfigMap = { ...value };
    clone[key] = updateAtPath(clone[key], rest, next);
    return clone;
  }
  throw new Error('Cannot update non-object path');
};

export const pathToString = (path: (string | number)[]): string =>
  path
    .map((segment) =>
      typeof segment === 'number' ? `[${segment}]` : path.length === 1 ? segment : `.${segment}`
    )
    .join('');
