export function buildNested(edits: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [dotted, value] of Object.entries(edits)) {
    const parts = dotted.split('.');
    let cursor: Record<string, unknown> = result;
    parts.slice(0, -1).forEach((part) => {
      if (!(part in cursor) || typeof cursor[part] !== 'object' || cursor[part] === null) {
        cursor[part] = {};
      }
      cursor = cursor[part] as Record<string, unknown>;
    });
    cursor[parts[parts.length - 1]] = value;
  }
  return result;
}
