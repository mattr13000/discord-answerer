"""Discord Answerer — RAG strictly bounded to the messages of an exported Discord.

Pipeline: JSON export (DiscordChatExporter) -> parse -> chunking into conversation
windows -> local embeddings -> numpy index -> cosine top-k search ->
bounded LLM synthesis (0 web, 0 assumption) with citations of the source messages.
"""

__version__ = "0.1.0"
