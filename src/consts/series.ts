export type SeriesKey = 'llm-fundamentals' | 'agent';

export const SERIES_CONFIG: Record<
	SeriesKey,
	{
		title: string;
		subtitle: string;
		description: string;
		dayRange: string;
	}
> = {
	'llm-fundamentals': {
		title: '大模型基础',
		subtitle: 'Foundation',
		description: 'Transformer、注意力机制、预训练与微调等底层架构知识',
		dayRange: 'Day 1 – 79',
	},
	agent: {
		title: 'Agent',
		subtitle: 'Intelligent Agents',
		description: '规划、ReAct、反思、工具调用与记忆等 Agent 核心能力',
		dayRange: 'Day 80 – 100',
	},
};

export const SERIES_ORDER: SeriesKey[] = ['llm-fundamentals', 'agent'];
