### MyRA: My Research Assistant
MyRA is a simple application for curating recent arXiv papers and chatting with ChatGPT for your research.
At this moment, MyRA is experimental so any comments or ideas for improvements are welcome. 


#### How it works
1. Retrieve recent arXiv papers whose category belongs to your input arXiv categories (e.g., cs.CV for computer vision).
2. Rank the papers by applying a term-frequency-based document ranker (i.e., BM25) to each of paper titles and abstracts.
3. Select top-k papers from the reranked papers.
4. Summarise abstract using a GPT model, which can be either gpt-3-text-davinci (less users, more stable connection to OpenAI) or gpt-3.5-turbo (i.e., one used for ChatGPT - more cost-efficient and powerful, but unstable connection due to excessive requests from many users).
5. Finally, ChatGPT will answer your question based on the summarised abstracts from above.

#### Limitations
Due to the limitation of a context length, we are currently feeding ChatGPT an abstract summary of each curated paper. Due to this, it may not be possible to discuss about a retrieved paper in depth. 

#### Contact
If you have any questions or suggestions, please contact: [gyungin[at]robots[dot]ox[dot]ac[dot]uk](removethisifyouarehuman-gyungin@robots.ox.ac.uk)
