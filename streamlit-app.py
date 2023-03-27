from typing import List, Optional
import json
import numpy as np
from tqdm import tqdm
import arxiv
from arxiv import Client, SortCriterion, SortOrder
from rank_bm25 import BM25Okapi
import openai
import tiktoken
import streamlit as st


def retrieve_from_arxiv(arxiv_ids: List[str], n_max_papers: int = 10000, sort_by=SortCriterion.SubmittedDate):
    assert sort_by in [SortCriterion.SubmittedDate, SortCriterion.LastUpdatedDate]

    client = Client(page_size=1000, delay_seconds=3, num_retries=3)

    # retrieve abstracts from papers for the given arxiv ids
    paper_info: List[dict] = list()
    titles: set = set()
    for arxiv_id in arxiv_ids:
        for result in client.results(arxiv.Search(
            query=arxiv_id,
            max_results=n_max_papers,
            sort_by=sort_by,
            sort_order=SortOrder.Descending,
        )):
            title: str = result.title
            abstract: str = result.summary
            authors = [author.name for author in result.authors]
            published_date = f"{result.published:'%Y_%m_%d'}"
            pdf_url = result.pdf_url

            if title not in titles:
                paper_info.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "published_date": published_date,
                    "pdf_url": pdf_url
                })
                titles.update(title)

                if len(titles) == n_max_papers:
                    break
    return paper_info


def rerank_documents(paper_info: List[dict], query: List[str]) -> np.ndarray:
    title_abstract_strings = [f"{paper['title']} {paper['abstract']}" for paper in paper_info]
    tokenized_corpus = [doc.split(" ") for doc in title_abstract_strings]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query: List[str] = " ".join(query).split(" ")

    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(-scores)

    return sorted_indices


def summarise_with_score(research_keywords: List[str], paper_info: List[dict], model_name: str) -> (List[str], List[int]):
    scores: List[int] = list()
    summaries: List[str] = list()
    for paper in tqdm(paper_info):
        abstract: str = paper["abstract"]

        content = "Summarise the following abstract the abstract:"
        content += f"\n{abstract}"

        if model_name == "gpt-3.5-turbo":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ]
            )
            summary: str = response['choices'][0]['message']['content']

        elif model_name == "text-davinci-003":
            completions = openai.Completion.create(
                model='text-davinci-003',
                prompt=content,
                max_tokens=512,
                temperature=0.7,
                stream=False
            )
            summary = completions.choices[0].text

        summaries.append(summary)

    return summaries, scores


# copy-pasted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    encoded_tokens = encoding.encode(string)
    n_tokens = len(encoded_tokens)
    return n_tokens, encoded_tokens


def estimate_reading_cost(paper_info: List[dict], research_keywords: List[str], model_name: str) -> float:
    research_keywords_string = ', '.join(research_keywords)
    for paper in paper_info:
        abstract: str = paper["abstract"]

        content = 'Summarise the following abstract and rate the abstract based on how relevant it is to %s in a score ranging from 1 to 100. ' \
                  'Answer in the strict form of a python dictionary {"score": {score}, "summary": {summary}}.' % research_keywords_string
        content += f"\n{abstract}"

    n_tokens, encoded_tokens = num_tokens_from_string(content, model_name)

    if model_name == "gpt-3.5-turbo":
        cost = 0.002 * n_tokens / 1000 * 2  # multiply 2 for generation tokens (generous estimate)
    elif model_name == "text-davinci-003":
        cost = 0.02 * n_tokens / 1000 * 2  # multiply 2 for generation tokens (generous estimate)
    else:
        raise ValueError(f"Model name ({model_name}) not recognised.")
    return cost


def generate_response(
        prompt,
        model_name: str,
        paper_info: Optional[List[dict]] = None,
        research_keywords: Optional[List[str]] = None,
):
    research_keywords_string = ', '.join(research_keywords)
    if model_name == "gpt-3.5-turbo":
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": f"My research interests are {research_keywords_string}."})

        if paper_info is not None:
            background_info = f"The below is information about recent papers regarding {research_keywords_string}.\n"
            for info in paper_info:
                title = info["title"]
                authors = ', '.join(info["authors"])
                published_date = info["published_date"].replace('_', ' ')
                summary = info["summary"]

                background_info += f"Title: {title}\nAuthors: {authors}\nPublished Date: {published_date}\nSummary: {summary}\n"

            messages.append({"role": "user", "content": background_info})

        messages.append({"role": "user", "content": f"Answer the following question strictly based on the information above:\n"})
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response['choices'][0]['message']['content']

    elif model_name == "text-davinci-003":
        content = f"My research interests are {research_keywords_string}.\n"

        if paper_info is not None:
            background_info = f"The below is information about recent papers regarding {research_keywords_string}.\n"
            for info in paper_info:
                title = info["title"]
                authors = ', '.join(info["authors"])
                published_date = info["published_date"].replace('_', ' ')
                summary = info["summary"]

                background_info += f"Title: {title}\nAuthors: {authors}\nPublished Date: {published_date}\nSummary: {summary}\n"
            content += background_info
        content += f"Answer the following question strictly based on the information above:\n"
        content += prompt

        completions = openai.Completion.create(
            model='text-davinci-003',
            prompt=content,
            max_tokens=512,
            temperature=0.7,
            stream=False
        )
        return completions.choices[0].text


if __name__ == '__main__':
    import os
    import yaml
    from time import time
    from datetime import date
    from streamlit_chat import message
    from dotenv import load_dotenv

    load_dotenv('api_key.env')

    # Check if 'key' already exists in session_state
    # If not, then initialize it
    if 'key' not in st.session_state:
        st.session_state['key'] = 'value'

    if 'json_string' not in st.session_state:
        st.session_state.json_string = ''

    st.title("MyRA: My Research Assistant")

    arxiv_id_to_subject_desc = yaml.load(open("arxiv_subjects.yaml", 'r'), Loader=yaml.BaseLoader)
    arxiv_ids = list(arxiv_id_to_subject_desc.keys())
    openai_api_key: str = st.text_input(
        label="OpenAI API key",
        help="If you don't have one yet, get it from https://platform.openai.com/account/api-keys",
        value=""
    )

    arxiv_categories = st.multiselect(
        label="arXiv categories",
        options=arxiv_ids,
        default="cs.AI",
        help="Select the arXiv categories (e.g., cs.CV for computer vision. You can check more at https://arxiv.org/category_taxonomy) you want to retrieve papers from. You can select multiple categories."
    )
    research_keywords: str = st.text_input(label="Research keywords (comma separated)", value="")
    research_keywords: List[str] = research_keywords.split(",")

    retrieved_paper_numbers = st.number_input(
        label="How many recent papers do you want to retrieve from arXiv?",
        help="This number means the number of papers retrieved from arXiv which will be ranked by MyRA based on the research keywords you entered before it actually reads them. The maximum value is 10000.",
        min_value=1,
        max_value=10000,
        value=1000
    )

    model_name = st.selectbox(
        "GPT reader model",
        options=["text-davinci-003", "gpt-3.5-turbo"],
        help="While gpt-3.5-turbo is more affordable and better in performance, it is sometimes not available due to excessive requests for this model to OpenAI. "
             "In such a case, we recommend text-davinci-003."
    )

    paper_numbers = st.slider(
        label="How many papers do you want MyRA to read?",
        min_value=1,
        max_value=30,
        value=5
    )

    read_button = st.button(label="Go read!")

    if read_button:
        if openai_api_key == "":
            st.error("Please enter your OpenAI API key.")

        elif arxiv_categories == []:
            st.error("Please select at least one arXiv category.")

        elif research_keywords == "":
            st.error("Please enter at least one research keyword.")

        else:
            # set the openai api key
            openai.api_key = openai_api_key

            # retrieve abstracts from papers for the given arxiv ids
            with st.spinner(f"Retrieving {retrieved_paper_numbers} papers... (can take up to a few mins.)"):
                start_time = time()
                paper_info = retrieve_from_arxiv(arxiv_categories, n_max_papers=retrieved_paper_numbers)
                st.write(f"{retrieved_paper_numbers} papers are retrieved from arXiv ({time() - start_time:.2f} sec.).")

            # rerank the papers based on terms in the research keywords
            start_time = time()
            sorted_indices: np.ndarray = rerank_documents(paper_info=paper_info, query=research_keywords)
            st.write(f"The retrieved papers are reranked based on the research keywords ({time() - start_time:.2f} sec.).")

            # select only top-k papers
            selected_paper_info = [paper_info[i] for i in sorted_indices[:paper_numbers]]

            # estimate the cost of reading the papers
            cost = estimate_reading_cost(
                paper_info=selected_paper_info, research_keywords=research_keywords, model_name=model_name
            )
            st.write(f"Estimated price for reading the {paper_numbers} papers is ${cost:.3f}.")

            # summarise the abstracts of the papers
            with st.spinner(f"Summarising top {paper_numbers} papers... (can take up to a few mins.)"):
                start_time = time()
                summaries, scores = summarise_with_score(
                    research_keywords=research_keywords, paper_info=selected_paper_info, model_name=model_name
                )
                st.write(f"Done ({time() - start_time:.3f} sec.).")

                # store the summaries and scores with the existing paper info
                for i in range(len(selected_paper_info)):
                    selected_paper_info[i]['summary'] = summaries[i]
                    selected_paper_info[i].pop("abstract")  # replace the abstract with the summary

                json_string = json.dumps(selected_paper_info, indent=4)

                st.session_state["json_string"] = json_string
                st.json(json_string, expanded=False)

                st.session_state["research_keywords"] = research_keywords
                st.session_state["selected_paper_info"] = selected_paper_info
                st.write("MyRA has finished reading the papers. You can start chatting with MyRA now!")

    # download the summaries
    st.download_button(
        label="Download summaries",
        data=st.session_state["json_string"],
        file_name=f'myra_{model_name}_{date.today().strftime("%Y_%m_%d")}.json',
    )

    # chatting with MyRA
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = st.text_input("You:", key='input')

    if user_input:
        if 'selected_paper_info' not in st.session_state:
            st.error(
                "You haven't made MyRA read any papers yet. "
                "This means MyRA might not be able to assist your research properly. "
                "Please click the 'Go read!' button first to make MyRA read some recent papers."
            )
            output = generate_response(prompt=user_input, paper_info=None, model_name="gpt-3.5-turbo")  # model_name)

        else:
            output = generate_response(
                prompt=user_input,
                paper_info=st.session_state["selected_paper_info"],
                research_keywords=st.session_state["research_keywords"],
                model_name="gpt-3.5-turbo"  # model_name
            )

        # store the output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
