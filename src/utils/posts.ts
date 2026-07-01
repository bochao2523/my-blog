import type { CollectionEntry } from 'astro:content';

export type BlogPost = CollectionEntry<'blog'>;

export function getDayNumber(id: string): number {
	const match = id.match(/^day(\d+)/);
	return match ? parseInt(match[1], 10) : 0;
}

export function sortByDay(posts: BlogPost[]): BlogPost[] {
	return [...posts].sort((a, b) => getDayNumber(a.id) - getDayNumber(b.id));
}
