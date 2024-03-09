from langchain.prompts import PromptTemplate


qa_template = """You are a member of UNOX Customer Service team. Your task is to help customers to find the product that suits them more among all the products available.
You have been provided with a user query, delimited by '#####'. The following pieces of context which is from a csv file, delimited by '-----', contains data of UNOX products. 

Your tasks are to:
1- You have recommend the best product(s) according to user's query using ONLY the information provided in the context.
2- If you cannot answer the question with the provided context, return an empty array.
3- Respond a JSON object formatted according to the following schema:


#####{query}#####

-----{context}-----

"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)


# Map-Reduce prompts
########################################################################
# MAP REDUCE
map_prompt="""You have been provided with some pages of a document delimited by '#####'. Your task is to carefully craft a structured summary that addresses potential questions. The summary should contain enough information so that one can determine on which page or page range further details about the question can be found.
To achieve this objective, follow the steps below, creating a section for each:
1. Identify and Output the Document's Metadata:
    Title: The title of the document.
    Author(s): Name of the author(s) or organization responsible for the document.
    Date of publication: Date that the document was published.
    Upload, and/or last modification: Date that the document was uploaded and/or last modified.
    Source/URL: If the document was sourced from a particular location or website.
    Total Pages (This chunk): Total number of pages in the input chunk.
    Total Pages (Document): Total number of pages in the document (Up to this chunk. Which is the last page of input).
    Uploader: Name of the user who uploaded the document.

2. Create a Table of Contents (ToC):
    - Identify sections, subsections, chapters, etc.
    - For sections that do not start or end within the chunk, mark them as 'Continued' or 'Begins' appropriately, indicating that the content extends beyond the current chunk.
    - If there were no new chapters or sections in this chunk, indicate that the chunk is continued from previous chunk. Specify this situation by writing 'Continue of previous chapter' in the ToC section.

3. Create Summaries and Key Points by Section:
    For each major section or chapter, provide:
        Section Title: Title of the section or chapter.
        Page Range: If the section begins within this chunk, note the starting page number of this section and the ending page of previous section. If the section continues into the next chunk, indicate this accordingly. 
        Subsections (if applicable): List major subsections and their starting page numbers.
        Summary Points: Bullet points summarizing the key details or findings from that section. For each summary point, provide the page number(s) where the information can be found.

#####{context}#####

REMEMBER, Ensure you return ONLY the summary, without any additional notes or explanations in your output.
REMEMBER, Check thoroughly the text to detect all the explicit sections and subsection titles (e.g., chapter, CHAPTER, etc.).
REMEMBER, It is IMPORTANT to include all the section and subsection titles in the ToC part.
REMEMBER, for sections that do not conclude within your pages, clearly indicate the starting point or continuation of such sections in the ToC and Summary Points.
REMEMBER, if the section spans multiple chunks, clearly mark it as 'Continues on next chunk' and provide the starting page number. If a new section starts, ensure to mark the previous section's ending page number.
"""

reduce_prompt="""You have been provided with several structured summaries of consecutive sections of a document. The summaries are delimited by '#####'. Each structured summary pertains to a fixed number of pages (e.g., 10 pages) of the original document. The structured summaries are also ordered according to the pages of the original document. Your task is to combine all these summaries into a single one, while maintaining the overall structure and the information within the summaries.
Using a metaphor, if you consider each structured summary as a set, you need to perform a union of all the structured summaries.

To achieve the objective, follow these steps:
1- Review the context and identify each structured summary.
2- Starting with the first structured summary, merge all the information from the second summary.
3- Begin with the third summary (if there is any) and merge it completely with the combined summary of the first and second summaries.
4- Continue this process until you reach the last summary.

For the ToC section:

1. Carefully Review Each Structured Summary:
   - Identify and note the ToC and page ranges provided in each summary.

2. Construct a Combined ToC:
   - For each chapter/section, you can find its start page.
   - You can find the ending of each chapter, by looking at the start page of the next chapter.

3. Merge the Summaries:
   - Integrate the summaries, aligning the content with the finalized ToC.
   - Ensure that the key points and summaries are placed under the correct ToC entries.

REMEMBER:
- It is crucial to maintain the integrity of the ToC, reflecting the exact order and page ranges of the document.
- Carefully review the extended summary and modify it as needed to ensure the ToC and summary points are integrated and organized in the correct order.
- Identify the page range for each chapter or section across the summaries. Last chapter will finish with the last page of the document.
- Merge the summaries, ensuring ToC and summary points from connected sections are correctly integrated.
- There are some chunks without any new chapter or section in them. You have to consider them as continuation of the previous chunk.
- Show the page range of chapters in ToC section in this format: Page (start page)-(end page). For instance, if chapter 3 starts in page 10 and ends in page 20, you have to write 'page 10-20' in the ToC section.

#####{context}#####

REMEMBER, it is IMPORTANT to include all the section and subsection titles in the ToC and Key-points parts.
REMEMBER, the main objective is to determine on which page or pages specific information can be found.
REMEMBER, ensure you return ONLY the extended structured summary, without any additional notes or explanations in your output.
"""

MAP_PROMPT = PromptTemplate.from_template(map_prompt)
REDUCE_PROMPT = PromptTemplate.from_template(reduce_prompt)

