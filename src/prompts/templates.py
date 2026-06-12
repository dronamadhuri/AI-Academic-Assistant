from langchain_core.prompts import PromptTemplate

# Prompt Template for academic QA / Retrieval-Augmented Generation
ACADEMIC_QA_TEMPLATE = """You are an expert academic research assistant. 
Use the following pieces of retrieved context from scientific publications to answer the question at the end.
Provide inline citations using author names or paper titles. If you do not know the answer, state that clearly.
Maintain a formal, objective, and rigorous academic tone.

Context:
{context}

Question: {question}

Answer:"""

ACADEMIC_QA_PROMPT = PromptTemplate(
    template=ACADEMIC_QA_TEMPLATE,
    input_variables=["context", "question"]
)

# Prompt Template for summarizing scientific text
SUMMARIZATION_TEMPLATE = """Synthesize and summarize the following scientific text. 
Focus on:
1. Research problem / Hypothesis
2. Core Methodology
3. Key Findings & Contributions

Text:
{text}

Summary:"""

SUMMARIZATION_PROMPT = PromptTemplate(
    template=SUMMARIZATION_TEMPLATE,
    input_variables=["text"]
)

# Structured Prompt Template for Summary Service
ACADEMIC_SUMMARY_TEMPLATE = """You are an expert academic summarization assistant.
Analyze the following scientific text and extract a comprehensive, structured summary.
Your response MUST strictly follow the markdown header structure below:

# EXECUTIVE SUMMARY
[Provide a clear, high-density academic summary paragraph outlining background, methodology, and key achievements.]

# KEY TOPICS
- [Topic 1]: Short explanation.
- [Topic 2]: Short explanation.
...

# IMPORTANT CONCEPTS
- [Concept 1]: Detailed explanation of the theory or concept and its role.
- [Concept 2]: Detailed explanation of the theory or concept and its role.
...

# IMPORTANT DEFINITIONS
- [Term 1]: Exact scientific definition.
- [Term 2]: Exact scientific definition.
...

Text Content:
{text}

Structured Summary:"""

ACADEMIC_SUMMARY_PROMPT = PromptTemplate(
    template=ACADEMIC_SUMMARY_TEMPLATE,
    input_variables=["text"]
)

# Structured Prompt Template for Assignment Generation Service
ASSIGNMENT_GENERATION_TEMPLATE = """You are an expert academic writer and professor.
Using the provided scientific text content, generate a comprehensive, highly-structured, and rigorous academic assignment.
The target length for this assignment is: {length_desc} (approximately {word_count} words).
Ensure you elaborate fully and write exhaustive, detailed content to satisfy this length requirement.

Your response MUST strictly follow the markdown header structure below:

# ASSIGNMENT TITLE
[Provide a professional, scholarly title for the assignment.]

# INTRODUCTION
[Write a comprehensive introduction outlining background, research questions, context, and assignment layout.]

# MAIN CONTENT
[Exhaustively develop the main scientific analysis, critiques of the methodologies, discussions, and details. Write rich, academic prose.]

# CONCLUSION
[Summarize key outcomes, limitations, contributions, and future directions.]

# REFERENCES
[Compile standard academic references/sources in APA format.]

Provided Scientific Text:
{text}

Academic Assignment:"""

ASSIGNMENT_GENERATION_PROMPT = PromptTemplate(
    template=ASSIGNMENT_GENERATION_TEMPLATE,
    input_variables=["text", "length_desc", "word_count"]
)

# Structured Prompt Template for Study Notes Generator Service
STUDY_NOTES_TEMPLATE = """You are an expert academic tutor.
Analyze the following scientific text and compile a set of comprehensive, clear study notes.
Your response MUST strictly follow the markdown header structure below:

# CHAPTER-WISE NOTES
[Provide detailed, chapter-wise or section-wise notes summarizing the main text. Break down complex topics into subheadings if applicable.]

# KEY CONCEPTS
- [Concept 1]: Clear explanation and significance.
- [Concept 2]: Clear explanation and significance.
...

# REVISION POINTS
- [Revision Point 1]: Concise, easy-to-remember core takeaway.
- [Revision Point 2]: Concise, easy-to-remember core takeaway.
...

# EXAM TIPS
- [Exam Tip 1]: High-probability test topic or common pitfall to avoid.
- [Exam Tip 2]: Typical exam question context or key focus.
...

Provided Scientific Text:
{text}

Study Notes:"""

STUDY_NOTES_PROMPT = PromptTemplate(
    template=STUDY_NOTES_TEMPLATE,
    input_variables=["text"]
)

# Structured Prompt Template for Viva Question Generator Service
VIVA_QUESTIONS_TEMPLATE = """You are an expert academic examiner conducting a viva voce (oral examination).
Analyze the following scientific text and generate exactly 20 questions for each of the three difficulty levels: Beginner, Intermediate, and Advanced.
For each question, provide a brief suggested answer or key evaluation points.

Your response MUST strictly adhere to the markdown header structure below:

# BEGINNER QUESTIONS
1. [Question]: [Suggested Answer / Key Points]
2. [Question]: [Suggested Answer / Key Points]
...
20. [Question]: [Suggested Answer / Key Points]

# INTERMEDIATE QUESTIONS
1. [Question]: [Suggested Answer / Key Points]
2. [Question]: [Suggested Answer / Key Points]
...
20. [Question]: [Suggested Answer / Key Points]

# ADVANCED QUESTIONS
1. [Question]: [Suggested Answer / Key Points]
2. [Question]: [Suggested Answer / Key Points]
...
20. [Question]: [Suggested Answer / Key Points]

Provided Scientific Text:
{text}

Viva Questions:"""

VIVA_QUESTIONS_PROMPT = PromptTemplate(
    template=VIVA_QUESTIONS_TEMPLATE,
    input_variables=["text"]
)

# Structured Prompt Template for MCQ Generator Service
MCQ_QUESTIONS_TEMPLATE = """You are an expert academic test generator.
Analyze the following scientific text and generate exactly 25 Multiple Choice Questions (MCQs).
Each question must contain:
1. Question text.
2. Exactly 4 options: (A), (B), (C), and (D).
3. The correct answer option indicator.
4. A concise scholarly explanation.

Your response MUST strictly adhere to the following tag format for each question:

QUESTION: [Write the question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
ANSWER: [A, B, C, or D]
EXPLANATION: [Write the explanation here]

Ensure there is a blank line between each question block. Do not include other conversational intro or outro text.

Provided Scientific Text:
{text}

Multiple Choice Questions:"""

MCQ_QUESTIONS_PROMPT = PromptTemplate(
    template=MCQ_QUESTIONS_TEMPLATE,
    input_variables=["text"]
)

# Structured Prompt Template for PPT Generator Service
PPT_GENERATION_TEMPLATE = """You are an expert academic presentation designer.
Analyze the following scientific text and generate the content for exactly 10 presentation slides.
Each slide must contain:
1. Slide Title
2. Bullet Points (key summaries)
3. Speaker Notes (detailed explanations for the presenter)

Your response MUST strictly adhere to the following tag format for each slide:

SLIDE_TITLE: [Write the slide title here]
BULLET_POINTS:
- [Point 1]
- [Point 2]
...
SPEAKER_NOTES: [Write the speaker notes here. Provide context, details, or a narrative for the slide presenter.]

Ensure there is a blank line between each slide block. Do not include other conversational intro or outro text.

Provided Scientific Text:
{text}

Presentation Slides:"""

PPT_GENERATION_PROMPT = PromptTemplate(
    template=PPT_GENERATION_TEMPLATE,
    input_variables=["text"]
)

# Structured Prompt Template for RAG Chat Service
CHAT_RAG_TEMPLATE = """You are a precise academic research assistant.
Answer the user's question based ONLY on the provided context retrieved from the uploaded research papers.
Strictly adhere to the following rules:
1. Answer the question using ONLY the facts directly mentioned in the context.
2. Do NOT extrapolate, speculate, or use external knowledge.
3. If the answer cannot be found in the provided context, you must answer EXACTLY with: "Information not found in uploaded document."

Provided Context:
{context}

User Question: {question}

Answer:"""

CHAT_RAG_PROMPT = PromptTemplate(
    template=CHAT_RAG_TEMPLATE,
    input_variables=["context", "question"]
)

# --- Research and Visual Enhancements ---

RESEARCH_MODE_MODIFIER = """
[RESEARCH MODE ACTIVE]
Apply formal peer-reviewed academic rigor. Critically analyze scientific methodologies, discuss limitations, theoretical structures, and research implications. Maintain an objective, high-density scientific prose format.
"""

MERMAID_DIAGRAM_TEMPLATE = """You are an expert academic visualizer.
Analyze the following scientific text and design a structural workflow, flowchart, or concept relationship map using Mermaid diagram syntax.
Your output must contain ONLY valid Mermaid diagram syntax enclosed within a ```mermaid code block.
Do not write any introductory or explanatory text.

Scientific Text:
{context}

Mermaid Diagram:"""

MERMAID_DIAGRAM_PROMPT = PromptTemplate(
    template=MERMAID_DIAGRAM_TEMPLATE,
    input_variables=["context"]
)







