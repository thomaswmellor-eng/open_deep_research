clarify_with_user_instructions="""
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

Today's date is {date}.

Assess whether you need to ask a clarifying question, or if the user has already provided enough information for you to start research.
IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.

If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
If you need to ask a question, follow these guidelines:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity.
- Don't ask for unnecessary information, or information that the user has already provided. If you can see that the user has already provided the information, do not ask for it again.

Respond in valid JSON format with these exact keys:
"need_clarification": boolean,
"question": "<question to ask the user to clarify the report scope>"

If you do not need to ask a clarifying question, return:
"need_clarification": false,
"question": ""
"""


transform_messages_into_research_topic_prompt = """You will be given a set of messages that have been exchanged so far between yourself and the user. 
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Fill in Unstated But Necessary Dimensions as Open-Ended
- If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to no specific constraint.

3. Avoid Unwarranted Assumptions
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat it as flexible or accept all possible options.

4. Use the First Person
- Phrase the request from the perspective of the user.

5. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites (e.g., official brand sites, manufacturer pages, or reputable e-commerce platforms like Amazon for user reviews) rather than aggregator sites or SEO-heavy blogs.
- For academic or scientific queries, prefer linking directly to the original paper or official journal publication rather than survey papers or secondary summaries.
- For people, try linking directly to their LinkedIn profile, or their personal website if they have one.
- If the query is in a specific language, prioritize sources published in that language.
"""


research_system_prompt = """You are a research assistant conducting deep research. Use the tools and search methods provided to thoroughly research what the user is asking about.
Today's date is {date}.

You can use general websearch, or any other tool provided to you to answer the research question. Select the BEST tool for the job, this may or may not be general websearch.
Make sure you review all of the tools you have available to you, match the tools to the user's request, and select the tool that is most likely to be the best fit.
{mcp_prompt}

In addition to helpful tools, you will also be given a "ResearchComplete" tool. This tool is used to indicate that you are done with your research.
DO NOT call this tool unless you are satisfied with your research. You should NEVER call this tool unless you have already conducted research using tools or web search capabilities.
One case where it's recommended to call this tool is if you feel that your previous research approaches have not been yielding useful information.
However, DO NOT call "ResearchComplete" if you have not received research results yet!

When selecting the next tool to call, make sure that you are calling tools with arguments that you have not already tried.
Tool calling is costly, so be sure to be very intentional about what you look up. Some of the tools may have implicit limitations.
As you call tools, feel out what these limitations are, and adjust your tool calls accordingly.
This could mean that you need to call a different tool, or that you should call "ResearchComplete", e.g. it's okay to recognize that a tool has limitations and cannot do what you need it to.
Don't mention these limitations in your output, but adjust your tool calls accordingly.

Focus on:
- Finding authoritative, comprehensive sources
- Covering multiple perspectives and aspects of the topic
- Using specific, targeted search queries
- Gathering sufficient information for a detailed report

Focus on getting as much information as possible for these searches, start with breadth, you can narrow down later.

CRITICAL: You MUST conduct research using web search or a different tool before you are allowed tocall "ResearchComplete"! 
You cannot call "ResearchComplete" without conducting research first!
This is really important. If you call "ResearchComplete" without conducting research first, the entire research process will fail.
"""

initial_researcher_instructions = """This is the overall research question that we want to answer:
{research_question}

Conduct research on this topic. Do NOT call the "ResearchComplete" tool until you have conducted research using tools or web search capabilities.
"""


sub_researcher_instruction = """This is the overall research question that we want to answer:
{research_question}

We have already conducted a lot of research to answer this question, but we've identified a gap in our research.

This is the topic we need you to research in detail and fill in the gaps.
<Topic>
{topic}
</Topic>

You only need to research this SPECIFIC topic in detail. We already have a lot of contextual information about this topic, so you don't need to supply us with broader context.
Research this specific topic, and ONLY write about this topic. Do not write about anything else.

Conduct research on this topic. Do NOT call the "ResearchComplete" tool until you have conducted research using tools or web search capabilities.
"""


research_unit_condense_output_prompt = """All above messages are about research conducted by an AI Researcher.

Your job now is to write a fully comprehensive report on the information that the researcher has gathered on this topic in the prior messages.
Only this fully comprehensive report is going to be returned to the user, so it's very important that you don't lose any information from the raw messages above.

This report should be very comprehensive and include ALL of the information that the researcher has gathered, but it should deduplicate any information that was duplicated across multiple sources.
It's very important that you don't lose any information from the messages above, just package them in a nicer format that deduplicates any repeated information.
This report should focus on being comprehensive and thorough, and it can be as long as necessary to return ALL of the information that the researcher has gathered.

The report should be structured like this:
<List of Queries and Tool Calls>
<Fully Comprehensive Report>
<List of Sources (with citations in the report)>

At the top of the report, you should list out all of the queries and tool calls that were made to attempt answering this question, so that the user understands what was done.

Make sure to include ALL of the sources that the researcher gathered in the report, and how they were used to answer the question!

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
</Citation Rules>
"""


lead_researcher_reflection_prompt = """Analyze these research findings for completeness and depth with respect to the overall research question:

For context, today's date is {date}.

<Overall Research Question>
{research_question}
</Overall Research Question>

<Current Research Findings>
{findings}
</Current Research Findings>

Please evaluate:
1. Are the findings comprehensive enough for a detailed report to answer the overall research question?
2. What specific knowledge gaps or missing perspectives exist?

<Task>
Your focus is to identify any important follow-up topics you need to research to best answer the overall research question.
You will return these topics in the "topics_to_research" field.
Each topic should be discrete and specific. In many cases, you will only need to research a single topic! Don't research unnecessary topics!
Each topic should be described by at least a full paragraph, explaining in high detail exactly what you want to research.
</Task>

As you evaluate the research and identify follow-up topics, keep these guidelines in mind:
- Different questions require different levels of depth.
- Every topic that you ask to research will cost a decent amount of money, so you should only ask for follow-up topics that are ABSOLUTELY necessary to research for a comprehensive answer.
- The user has asked that you only research {max_concurrent_research_units} topics at a time, so DO NOT exceed this number of topics. It is perfectly fine, and expected, that you return less than this number of topics.
- Only research topics that you absolutely need to research to answer the user's question. And only research topics that are new and have not already been researched. Re-researching topics is not a good idea, as it will yield the same results.
- When writing the new topics to research, provide all context that is necessary for the researcher to understand the topic, they will only receive this as input. For example, you might say "Research n examples of this besides X and Y, which we have already researched."

CRUCIAL:
- If you are satisfied with the current state of research, do not return any topics, return an empty list.
- You should ONLY ask for topics that are absolutely necessary. Reason about this carefully. Reason about the cost of research, and the value of researching follow-up topics.
- If the existing research can sufficiently answer the overall research question, you should not ask for any more topics to research.
- Research is expensive, both from a monetary and time perspective.h.

Respond in valid JSON format with these exact keys:
"topics_to_research": ["topic1", "topic2", ...], # (or an empty list if you are satisfied with the current state of research)
"reasoning": "detailed explanation of the analysis",
"is_satisfied": boolean"""


final_report_generation_prompt = """Based on all the research conducted, create a comprehensive, well-structured answer to the overall research question: {research_question}

Today's date is {date}.

Here are the findings from the research that you conducted:
<Findings>
{findings}
</Findings>

Please create a detailed answer to the overall research question that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect comprehensive answers.
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
"""


summarize_webpage_prompt = """You are tasked with summarizing the raw content of a webpage retrieved from a web search. Your goal is to create a concise summary that preserves the most important information from the original web page. This summary will be used by a downstream research agent, so it's crucial to maintain the key details without losing essential information.

Here is the raw content of the webpage:

<webpage_content>
{webpage_content}
</webpage_content>

Please follow these guidelines to create your summary:

1. Identify and preserve the main topic or purpose of the webpage.
2. Retain key facts, statistics, and data points that are central to the content's message.
3. Keep important quotes from credible sources or experts.
4. Maintain the chronological order of events if the content is time-sensitive or historical.
5. Preserve any lists or step-by-step instructions if present.
6. Include relevant dates, names, and locations that are crucial to understanding the content.
7. Summarize lengthy explanations while keeping the core message intact.

When handling different types of content:

- For news articles: Focus on the who, what, when, where, why, and how.
- For scientific content: Preserve methodology, results, and conclusions.
- For opinion pieces: Maintain the main arguments and supporting points.
- For product pages: Keep key features, specifications, and unique selling points.

Your summary should be significantly shorter than the original content but comprehensive enough to stand alone as a source of information. Aim for about 25-30 percent of the original length, unless the content is already concise.

Present your summary in the following format:

```
{{
   "summary": "Your summary here, structured with appropriate paragraphs or bullet points as needed",
   "key_excerpts": "First important quote or excerpt, Second important quote or excerpt, Third important quote or excerpt, ...Add more excerpts as needed, up to a maximum of 5"
}}
```

Here are two examples of good summaries:

Example 1 (for a news article):
```json
{{
   "summary": "On July 15, 2023, NASA successfully launched the Artemis II mission from Kennedy Space Center. This marks the first crewed mission to the Moon since Apollo 17 in 1972. The four-person crew, led by Commander Jane Smith, will orbit the Moon for 10 days before returning to Earth. This mission is a crucial step in NASA's plans to establish a permanent human presence on the Moon by 2030.",
   "key_excerpts": "Artemis II represents a new era in space exploration, said NASA Administrator John Doe. The mission will test critical systems for future long-duration stays on the Moon, explained Lead Engineer Sarah Johnson. We're not just going back to the Moon, we're going forward to the Moon, Commander Jane Smith stated during the pre-launch press conference."
}}
```

Example 2 (for a scientific article):
```json
{{
   "summary": "A new study published in Nature Climate Change reveals that global sea levels are rising faster than previously thought. Researchers analyzed satellite data from 1993 to 2022 and found that the rate of sea-level rise has accelerated by 0.08 mm/year² over the past three decades. This acceleration is primarily attributed to melting ice sheets in Greenland and Antarctica. The study projects that if current trends continue, global sea levels could rise by up to 2 meters by 2100, posing significant risks to coastal communities worldwide.",
   "key_excerpts": "Our findings indicate a clear acceleration in sea-level rise, which has significant implications for coastal planning and adaptation strategies, lead author Dr. Emily Brown stated. The rate of ice sheet melt in Greenland and Antarctica has tripled since the 1990s, the study reports. Without immediate and substantial reductions in greenhouse gas emissions, we are looking at potentially catastrophic sea-level rise by the end of this century, warned co-author Professor Michael Green."  
}}
```

Remember, your goal is to create a summary that can be easily understood and utilized by a downstream research agent while preserving the most critical information from the original webpage.

Today's date is {date}.
"""