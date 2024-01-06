import json
import requests
import numpy as np
import os
from django.http import JsonResponse, HttpResponseRedirect
from django.views.decorators.http import require_http_methods
from csas.models import DynamicIssueQuestion
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
from django.urls import reverse
from django.shortcuts import render, redirect, reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from csas.embeds import generate_embedding, cosine_similarity_knn, find_similar_issues, has_high_similarity, is_text_toxic, is_text_toxic_alt, has_high_similarity_alt, find_similar_issues_alt, generate_embedding_alt

def view_db_content(request):
    questions = DynamicIssueQuestion.objects.all()
    return render(request, 'view_db_content.html', {'questions': questions})

@csrf_exempt
def save_session_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        # Update session data
        request.session['prompt'] = data.get('prompt')
        request.session['ai_choice'] = data.get('ai_choice')
        request.session['num_items'] = data.get('num_items')

        # Create a dictionary of the session data
        session_data = {
            'prompt': request.session.get('prompt'),
            'ai_choice': request.session.get('ai_choice'),
            'num_items': request.session.get('num_items')
        }

        # Return both received and session data in the response
        return JsonResponse({
            'status': 'success',
            'message': 'Session data saved',
            'received_data': data,
            'session_data': session_data
        })

    else:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid request'
        },
                            status=400)


@csrf_exempt
@require_http_methods(["POST"])
def upload_completion(request):
    completion = request.POST.get('completion')
    ai_choice = request.POST.get('ai_choice')
    issue = request.POST.get('issue')

    if completion:
        # Save the completion to the database
        if ai_choice == 'mistral':
            global has_high_similarity
            global is_text_toxic
            has_high_similarity = has_high_similarity_alt
            is_text_toxic = is_text_toxic_alt

        # Check for an existing question in the database
        existing_question = DynamicIssueQuestion.objects.filter(
            question=completion).first()

        if existing_question or has_high_similarity(
                completion) or is_text_toxic(completion):
            # We will reuse the update_rating logic here, you'll likely want to abstract this logic into a common function
            new_rating = request.POST.get(
                'rating', 3)  # Default to 3 if no rating is provided
            new_rating = float(new_rating)

            # Update the existing question rating
            existing_question.ratings += 1
            existing_question.avg_rating = (
                (existing_question.avg_rating *
                 (existing_question.ratings - 1)) +
                new_rating) / existing_question.ratings
            existing_question.var_rating = 1 / existing_question.ratings
            existing_question.save()  # Don't forget to save the changes
        else:

            if not has_high_similarity(completion) and not is_text_toxic(
                    completion):
                embed_completion = generate_embedding(completion)
                # No existing question found and it's not similar or toxic, create a new one
                new_question = DynamicIssueQuestion(question=completion,
                                                    avg_rating=3,
                                                    ratings=1,
                                                    var_rating=1,
                                                    embedding=embed_completion)
                new_question.save()
            else:
                pass

        return redirect('view_db_content')

    return redirect('fetch_mistral_completion', issue=issue)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def main_page(request):
    context = {}

    if request.method == 'POST':
        issue = request.POST.get('issue')
        prompt = request.POST.get('prompt')
        ai_choice = request.POST.get('ai_choice')

        if not prompt:
            prompt = f"You are a classification expert who takes an input and returns a political issue as a summary. Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######"

        if ai_choice == 'openai':
            return redirect('fetch_openai_completion',
                            issue=issue,
                            prompt=prompt)
        elif ai_choice == 'mistral':
            return redirect('fetch_mistral_completion',
                            issue=issue,
                            prompt=prompt)

    elif request.method == 'GET':

        # Update the URLs to include the prompt
        dynamic_issue_url_oai = request.build_absolute_uri(
            '/dynamic-issue-oai/')
        dynamic_issue_url_mistral = request.build_absolute_uri(
            '/dynamic-issue-mistral/')
        select_questions_simulation_url = request.build_absolute_uri(
            reverse('select_questions_simulation'))
        update_rating_url = request.build_absolute_uri(
            reverse('update_rating'))

        context = {
            'dynamic_issue_url_oai': dynamic_issue_url_oai,
            'dynamic_issue_url_mistral': dynamic_issue_url_mistral,
            'select_questions_simulation_url': select_questions_simulation_url,
            'update_rating_url': update_rating_url
        }

    return render(request, 'main.html', context)


@csrf_exempt
@require_http_methods(["GET"])
def fetch_mistral_completion(
    request,
    issue,
    prompt="You are a classification expert who takes an input and returns a political issue as a summary. Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes. You must only return an issue. Keep it short.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######"
):

    try:
        previous_list = ', '.join(find_similar_issues_alt(issue))
    except:
        previous_list = ''

    data = {
        'model':
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'messages': [{
            'role':
            'user',
            'content':
            f"{prompt}Previously Mentioned Issues ({previous_list}) {issue}->"
        }],
        'temperature':
        0,
        'max_tokens':
        10
    }

    response = requests.Session().post(
        "https://api.endpoints.anyscale.com/v1/chat/completions",
        headers={"Authorization": f"Bearer " + os.environ["ANYSCALE_API_KEY"]},
        json=data)

    completion = response.json().get('choices',
                                     [{}])[0].get('message',
                                                  {}).get('content', '')
    completion = completion.split('#')[0].split('\n')[0].split(
        'Previously')[0].split('/')[0].split(';')[0]

    # Remove leading white space
    completion = completion.lstrip()

    return render(request, 'show_completion.html', {
        'completion': completion,
        'issue': issue
    })


@csrf_exempt
@require_http_methods(["GET"])
def fetch_openai_completion(
    request,
    issue,
    prompt="Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######Previously Mentioned Issues () "
):
    try:
        previous_list = ', '.join(find_similar_issues(issue))
    except:
        previous_list = ''

    data = {
        'model':
        'gpt-4',
        'messages': [{
            'role':
            'system',
            'content':
            f"Previously Mentioned Issues ({previous_list}) {prompt}{issue}->"
        }],
        'temperature':
        0,
        'max_tokens':
        15
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'
    }

    response = requests.post('https://api.openai.com/v1/chat/completions',
                             headers=headers,
                             data=json.dumps(data))

    completion = response.json().get('choices',
                                     [{}])[0].get('message',
                                                  {}).get('content', '')

    return render(request, 'show_completion.html', {
        'completion': completion,
        'issue': issue
    })


@require_http_methods(["GET"])
def dynamic_issue_openai(request):
    issue = request.GET.get('input')

    prompt_template = "Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes. If an issue is not political or irrelevant, return Room Temperature Semiconductors.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######"

    try:
        previous_list = ', '.join(find_similar_issues(issue))
    except:
        previous_list = ''

    prompt = request.GET.get('prompt', prompt_template)

    if prompt != prompt_template:
        # Format the prompt for the chat ML
        prompt = [{
            'role': 'system',
            'content': prompt
        }, {
            'role':
            'user',
            'content':
            f"Previously Mentioned Issues ({previous_list}) {issue}->"
        }]

    data = {
        'model':
        'gpt-4',
        'messages': [{
            'role':
            'system',
            'content':
            f"Previously Mentioned Issues ({previous_list}) {prompt}{issue}->"
        }],
        'temperature':
        0,
        'max_tokens':
        15
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'
    }

    response = requests.post('https://api.openai.com/v1/chat/completions',
                             headers=headers,
                             data=json.dumps(data))
    completion = response.json().get('choices',
                                     [{}])[0].get('message',
                                                  {}).get('content', '')
    embed_completion = generate_embedding(completion)

    # Check if the completion is toxic, similar, or identical
    if is_text_toxic(completion) or has_high_similarity(completion):
        # Return a random question from the database as the completion
        random_question = DynamicIssueQuestion.objects.order_by('?').first()
        if random_question:
            completion = random_question.question
    else:
        # Set default ratings for the new question
        dynamic_issue_question = DynamicIssueQuestion(
            question=completion,
            avg_rating=3,
            ratings=1,
            var_rating=1,
            embedding=embed_completion)
        dynamic_issue_question.save()  # Save the instance to the database

    return JsonResponse({'completion': completion})


@require_http_methods(["GET"])
def dynamic_issue_mistral(request):
    issue = request.GET.get('input')

    try:
        previous_list = ', '.join(find_similar_issues_alt(issue))
    except:
        previous_list = ''

    prompt_template = f"You are a classification expert who takes an input and returns a political issue as a summary. Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes. If an issue is not political or irrelevant, return Room Temperature Semiconductors.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######Previously Mentioned Issues ({previous_list}) {issue}->"

    prompt = request.GET.get('prompt', prompt_template)

    data = {
        'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'messages': [{
            'role': 'user',
            'content': prompt
        }],
        'temperature': 0,
        'max_tokens': 10
    }

    response = requests.Session().post(
        "https://api.endpoints.anyscale.com/v1/chat/completions",
        headers={"Authorization": f"Bearer " + os.environ["ANYSCALE_API_KEY"]},
        json=data)

    completion = response.json().get('choices',
                                     [{}])[0].get('message',
                                                  {}).get('content', '')

    completion = completion.strip().split('#')[0].strip().split('\n')[0].strip(
    ).split('Previously')[0].strip().split('/')[0].strip().split(';')[0].strip(
    ).split(',')[0].strip().split('&')[0].strip().split('and')[0].strip()
    embed_completion = generate_embedding_alt(completion)

    # Check if the completion is toxic, similar, or identical
    if is_text_toxic_alt(completion) or has_high_similarity_alt(completion):
        # Return a random question from the database as the completion
        random_question = DynamicIssueQuestion.objects.order_by('?').first()
        if random_question:
            completion = random_question.question
    else:
        # Set default ratings for the new question
        dynamic_issue_question = DynamicIssueQuestion(
            question=completion,
            avg_rating=3,
            ratings=1,
            var_rating=1,
            embedding=embed_completion)
        dynamic_issue_question.save()  # Save the instance to the database

    return JsonResponse({'completion': completion})


@csrf_exempt
@require_http_methods(["GET"])
def update_rating(request):
    try:
        # Check if there are any query parameters
        if not request.GET:
            return JsonResponse(
                {
                    'status': 'error',
                    'message': 'No data provided.'
                },
                status=400)

        for question, new_rating_str in request.GET.items():
            new_rating = float(new_rating_str) if new_rating_str else 0
            if new_rating <= 0:
                continue  # Skip invalid ratings

            # Update rating for each question
            try:
                issue = DynamicIssueQuestion.objects.get(question=question)
                issue.ratings += 1
                issue.avg_rating = (
                    (issue.avg_rating *
                     (issue.ratings - 1)) + new_rating) / issue.ratings
                issue.var_rating = 1 / issue.ratings
                issue.save()
            except DynamicIssueQuestion.DoesNotExist:
                continue  # Skip if question not found

        return JsonResponse({
            'status': 'success',
            'message': 'Ratings updated successfully.'
        })

    except ValueError:
        return JsonResponse(
            {
                'status': 'error',
                'message': 'Invalid rating format.'
            },
            status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["GET"])
def view_issues(request):
    questions = DynamicIssueQuestion.objects.all()
    questions_json = serializers.serialize('json', questions)
    return JsonResponse(questions_json, safe=False)


def select_questions_simulation(request):
    # Getting all questions from the database
    questions = DynamicIssueQuestion.objects.all().values(
        'question', 'avg_rating', 'ratings', 'var_rating')
    # Convert database rows to means and covariances in numpy arrays
    means = []
    covariances = []
    for question in questions:
        means.append([question['avg_rating']])
        covariances.append(np.diag([question['var_rating']]))

    # Run 250 simulations
    num_sims = 250
    results = []
    for _ in range(num_sims):
        samples = [
            np.random.multivariate_normal(mean, cov)
            for mean, cov in zip(means, covariances)
        ]
        max_value_index = np.argmax(samples, axis=0)[0]
        results.append(max_value_index)

    # Count the number of times each index had the highest value
    counts = np.bincount(results)
    proportions = counts / num_sims

    # Applying probability floor and renormalizing
    floor = 0.01
    proportions = np.maximum(proportions, floor)
    proportions /= proportions.sum()

    num_items_str = request.session.get('num_items')
    print(num_items_str)
    # Set a default value if num_items is not found or if it's None
    num_items = int(num_items_str) if num_items_str is not None else 3
    # Randomly select K questions based on the proportions
    selected_indices = np.random.choice(len(questions),
                                        size=num_items,
                                        replace=False,
                                        p=proportions)

    # Convert numpy.int64 indices to Python int
    selected_indices = [int(i) for i in selected_indices]

    selected_questions = {}

    # Populate the dictionary with question and probability pairs
    for idx, i in enumerate(selected_indices, start=1):
        selected_questions[f'q_{idx}'] = questions[i]['question']
        selected_questions[f'pr_{idx}'] = proportions[i]

    return JsonResponse(selected_questions, safe=False)
