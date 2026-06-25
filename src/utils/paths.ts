export function withBase(path: string = '/'): string {
	const base = import.meta.env.BASE_URL;
	const normalizedBase = base.endsWith('/') ? base : `${base}/`;

	if (path === '/') {
		return base.endsWith('/') ? base.slice(0, -1) || '/' : base;
	}

	return `${normalizedBase}${path.replace(/^\//, '')}`;
}
