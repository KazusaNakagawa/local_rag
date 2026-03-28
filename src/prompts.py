from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "会話履歴と最新の質問を踏まえ、履歴なしでも意味が通じる単独の質問に言い換えてください。"
        "言い換えが不要な場合はそのまま返してください。",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """以下のコンテキストを参考に、質問に日本語で答えてください。

ルール:
- コンテキストに直接的な答えがある場合はそれを使って答えてください。
- 直接的な答えがなくても、関連する情報があれば「直接的な記録はありませんが、関連するメモとして〜」のように共有してください。
- コンテキストに全く関連する情報がない場合のみ「ノートに該当する情報が見つかりませんでした」と答えてください。
- 会話履歴がある場合は、前の回答を踏まえて自然に答えてください。

コンテキスト:
{context}""",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
