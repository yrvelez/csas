import requests
import os
from csas.models import DynamicIssueQuestion
import numpy as np

def generate_embedding(text):
    headers = {
        'Authorization': f'Bearer ' + os.environ['OPENAI_API_KEY'],
        'Content-Type': 'application/json'
    }
    data = {'input': text, 'model': 'text-embedding-ada-002'}
    response = requests.post('https://api.openai.com/v1/embeddings',
                             headers=headers,
                             json=data)
    return response.json().get('data')[0].get('embedding')


def generate_embedding_alt(text):
    headers = {
        'Authorization': f'Bearer ' + os.environ['ANYSCALE_API_KEY'],
        'Content-Type': 'application/json'
    }
    data = {'input': text, 'model': 'thenlper/gte-large'}
    response = requests.post(
        'https://api.endpoints.anyscale.com/v1/embeddings',
        headers=headers,
        json=data)
    return response.json().get('data')[0].get('embedding')


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm_vec2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    return dot_product / np.dot(norm_vec1, norm_vec2.T)


def cosine_similarity_knn(embedding, all_embeddings, top_n=3):
    if not all_embeddings:
        return None
    embedding_array_2d = np.array(embedding).reshape(1, -1)
    all_embeddings_array = np.array(all_embeddings)
    similarities = cosine_similarity(embedding_array_2d,
                                     all_embeddings_array).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    nearest_neighbors = [(index, similarities[index]) for index in top_indices]
    return nearest_neighbors


def find_similar_issues(text):
    embedding = generate_embedding(text)
    all_issue_questions = DynamicIssueQuestion.objects.all().values_list(
        'embedding', flat=True)
    if not all_issue_questions:
        return None
    # Ensure embeddings are in the correct format (list of lists)
    all_embeddings = [
        e if isinstance(e, list) else [] for e in all_issue_questions
    ]
    top_n_similar = cosine_similarity_knn(embedding, all_embeddings, top_n=3)
    similar_issues = []
    for idx, _ in top_n_similar:
        try:
            question = DynamicIssueQuestion.objects.get(pk=idx + 1).question
            similar_issues.append(question)
        except DynamicIssueQuestion.DoesNotExist:
            continue  # Skip if the corresponding question does not exist
    return similar_issues


def has_high_similarity(text):
    embedding = generate_embedding(text)
    all_issue_questions = DynamicIssueQuestion.objects.all().values_list(
        'embedding', flat=True)
    if not all_issue_questions:
        return False
    all_embeddings = [
        e if isinstance(e, list) else [] for e in all_issue_questions
    ]
    top_n_similar = cosine_similarity_knn(embedding, all_embeddings, top_n=3)
    for _, similarity in top_n_similar:
        if similarity > 0.95:
            return True
    return False


def has_high_similarity_alt(text):
    embedding = generate_embedding_alt(text)
    all_issue_questions = DynamicIssueQuestion.objects.all().values_list(
        'embedding', flat=True)
    if not all_issue_questions:
        return False
    all_embeddings = [
        e if isinstance(e, list) else [] for e in all_issue_questions
    ]
    top_n_similar = cosine_similarity_knn(embedding, all_embeddings, top_n=3)
    for _, similarity in top_n_similar:
        if similarity > 0.95:
            return True
    return False


def find_similar_issues_alt(text):
    embedding = generate_embedding_alt(text)
    all_issue_questions = DynamicIssueQuestion.objects.all().values_list(
        'embedding', flat=True)
    if not all_issue_questions:
        return None
    # Ensure embeddings are in the correct format (list of lists)
    all_embeddings = [
        e if isinstance(e, list) else [] for e in all_issue_questions
    ]
    top_n_similar = cosine_similarity_knn(embedding, all_embeddings, top_n=3)
    similar_issues = []
    for idx, _ in top_n_similar:
        try:
            question = DynamicIssueQuestion.objects.get(pk=idx + 1).question
            similar_issues.append(question)
        except DynamicIssueQuestion.DoesNotExist:
            continue  # Skip if the corresponding question does not exist
    return similar_issues


def is_text_toxic(text):
    headers = {
        'Authorization': f'Bearer ' + os.environ['OPENAI_API_KEY'],
        'Content-Type': 'application/json'
    }
    data = {'input': text}
    response = requests.post('https://api.openai.com/v1/moderations',
                             headers=headers,
                             json=data)
    moderation_result = response.json()
    return moderation_result.get('results')[0].get('flagged')


# Create open-source toxicity routine
from string import Template

B_INST, E_INST = "[INST]", "[/INST]"

PROMPT_TEMPLATE = Template(
    """Task: Check if there is unsafe content in '$agent_type' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violence and Hate. 
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
-  Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content. 
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning. 
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons. 
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances. 
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm. 
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
<END UNSAFE CONTENT CATEGORIES>
​
<BEGIN CONVERSATION>

$prompt

<END CONVERSATION>

Provide your safety assessment for $agent_type in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories."""
)


def format_prompt(role, message):
    prompt = PROMPT_TEMPLATE.substitute(prompt=message, agent_type=role)
    prompt = f"<s>{B_INST} {prompt.strip()} {E_INST}"
    return prompt


def is_text_toxic_alt(text):

    data = {
        'model': "Meta-Llama/Llama-Guard-7b",
        'prompt': format_prompt("User", text)
    }

    headers = {
        'Authorization': f'Bearer ' + os.environ['ANYSCALE_API_KEY'],
        'Content-Type': 'application/json'
    }

    response = requests.post(
        'https://api.endpoints.anyscale.com/v1/completions',
        headers=headers,
        json=data)

    completion = response.json().get('choices')[0].get('text')

    if 'unsafe' in completion:
        return True
    else:
        return False
