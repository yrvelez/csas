import json
import requests
import numpy as np
import os
from django.http import JsonResponse, HttpResponseRedirect
from django.views.decorators.http import require_http_methods
from .models import DynamicIssueQuestion, UserDatabase, GlobalSetting
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
from django.urls import reverse
from django.shortcuts import render, redirect, reverse
from django.contrib import admin
from .models import GlobalSetting
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .embeds import generate_embedding, cosine_similarity_knn, find_similar_issues, has_high_similarity, is_text_toxic, is_text_toxic_alt, has_high_similarity_alt, find_similar_issues_alt, generate_embedding_alt
import logging


def get_global_setting(key):
    try:
        return GlobalSetting.objects.get(key=key).value
    except GlobalSetting.DoesNotExist:
        return None


def view_db_content(request):
    questions = DynamicIssueQuestion.objects.all()
    return render(request, 'view_db_content.html', {'questions': questions})


def view_user_content(request):
    user_data = UserDatabase.objects.all()
    return render(request, 'view_user_content.html', {'user_data': user_data})


@csrf_exempt
@require_http_methods(["POST"])
def save_session_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Iterate over each key-value pair in the received data
            for key, value in data.items():
                # Update or create the global setting
                GlobalSetting.objects.update_or_create(
                    key=key, defaults={'value': value})

            return JsonResponse({
                'status': 'success',
                'message': 'Global settings updated successfully',
                'received_data': data
            })
        except json.JSONDecodeError:
            return JsonResponse(
                {
                    'status': 'error',
                    'message': 'Invalid JSON format'
                },
                status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            },
                                status=500)

    else:
        return JsonResponse(
            {
                'status': 'error',
                'message': 'Invalid request method'
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

        # Get midpoint of the scale from session
        midpoint = (float(get_global_setting('min_scale')) +
                    float(get_global_setting('max_scale'))) / 2

        if existing_question or has_high_similarity(
                completion) or is_text_toxic(completion):
            # We will reuse the update_rating logic here, you'll likely want to abstract this logic into a common function
            new_rating = request.POST.get(
                'rating',
                midpoint)  # Default to midpoint if no rating is provided
            new_rating = float(new_rating)

            # Update the existing question rating
            existing_question.ratings += 1
            existing_question.avg_rating = (
                (existing_question.avg_rating *
                 (existing_question.ratings - 1)) +
                new_rating) / existing_question.ratings
            existing_question.var_rating = (1 / existing_question.ratings)
            existing_question.save()  # Don't forget to save the changes
        else:

            if not has_high_similarity(completion) and not is_text_toxic(
                    completion):
                embed_completion = generate_embedding(completion)
                # No existing question found and it's not similar or toxic, create a new one
                new_question = DynamicIssueQuestion(question=completion,
                                                    avg_rating=midpoint,
                                                    ratings=1,
                                                    var_rating=1,
                                                    embedding=embed_completion)
                new_question.save()
            else:
                pass

        return redirect('view_db_content')

    return redirect('fetch_mistral_completion', issue=issue)


@csrf_exempt
@require_http_methods(["POST"])
def upload_completions(request):
    completions = request.POST.getlist('approved_completions')
    ai_choice = request.POST.get('ai_choice')

    # Get midpoint
    midpoint = (float(get_global_setting('min_scale')) +
                float(get_global_setting('max_scale'))) / 2

    # If midpoint is empty, set it to 3
    if not midpoint:
        midpoint = 3

    new_rating = float(request.POST.get(
        'rating', midpoint))  # Get the rating outside the loop

    if completions:
        # Select the appropriate functions based on ai_choice
        similarity_check = has_high_similarity_alt if ai_choice == 'mistral' else has_high_similarity
        toxicity_check = is_text_toxic_alt if ai_choice == 'mistral' else is_text_toxic

        for completion in completions:
            try:
                existing_question = DynamicIssueQuestion.objects.filter(
                    question=completion).first()

                if existing_question:
                    # Update the existing question rating
                    #update_existing_question_rating(existing_question,
                    #new_rating)
                    pass
                elif not similarity_check(completion) and not toxicity_check(
                        completion):
                    embed_completion = generate_embedding(completion)
                    # Create a new question
                    new_question = DynamicIssueQuestion(
                        question=completion,
                        avg_rating=new_rating,
                        ratings=1,
                        var_rating=1,
                        embedding=embed_completion)
                    new_question.save()
                # If completion is similar or toxic, no action is taken
            except Exception as e:
                # Log the exception or handle it as needed
                print(f"Error processing completion '{completion}': {str(e)}")

        return redirect('view_db_content')

    return redirect('view_db_content')


def update_existing_question_rating(question, new_rating):
    question.ratings += 1
    question.avg_rating = (
        (question.avg_rating *
         (question.ratings - 1)) + new_rating) / question.ratings
    question.var_rating = (1 / question.ratings)
    question.save()


@csrf_exempt
@require_http_methods(["GET", "POST"])
def main_page(request):
    context = {}

    # Fetch global settings
    settings = GlobalSetting.objects.all()
    global_settings = {setting.key: setting.value for setting in settings}

    # Use global settings or default values
    session_data = {
        'prompt': global_settings.get('prompt', 'Default Prompt'),
        'ai_choice': global_settings.get('ai_choice', 'openai'),
        'num_items': global_settings.get('num_items', '3'),
        'min_scale': global_settings.get('min_scale', '1'),
        'max_scale': global_settings.get('max_scale', '5'),
        'survey_text': global_settings.get('survey_text',
                                           'Rate on a 1-5 scale.')
    }

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
            '/dynamic-issue-oai/').replace('http:', 'https:')
        dynamic_issue_url_mistral = request.build_absolute_uri(
            '/dynamic-issue-mistral/').replace('http:', 'https:')
        select_questions_simulation_url = request.build_absolute_uri(
            reverse('select_questions_simulation')).replace('http:', 'https:')
        update_rating_url = request.build_absolute_uri(
            reverse('update_rating')).replace('http:', 'https:')
        ez_url = request.build_absolute_uri(reverse('survey')).replace(
            'http:', 'https:').replace(
                'survey/', 'survey') + '?id=${e://Field/ResponseID}'

        # Add embed iframe for ez_url
        ez_url = f'<iframe src="{ez_url}" width="500" height="500" frameborder="0" style="border:0" allowfullscreen></iframe>'

        context = {
            'dynamic_issue_url_oai': dynamic_issue_url_oai,
            'dynamic_issue_url_mistral': dynamic_issue_url_mistral,
            'select_questions_simulation_url': select_questions_simulation_url,
            'update_rating_url': update_rating_url,
            'ez_url': ez_url
        }

    return render(request, 'main.html', context)


@csrf_exempt
@require_http_methods(["GET"])
def fetch_mistral_completion(
    request,
    issue,
    prompt="You are a classification expert who takes an input and returns a political issue as a summary. Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes. You must only return an issue. Keep it short.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######"
):
    issues_list = issue.split(';')  # Split the issue string into a list
    completions = []  # List to hold completions for each issue

    for single_issue in issues_list:
        try:
            previous_list = ', '.join(find_similar_issues_alt(single_issue))
        except:
            previous_list = ''

        data = {
            'model':
            'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'messages': [{
                'role':
                'user',
                'content':
                f"{prompt}Previously Mentioned Issues ({previous_list}) {single_issue}->"
            }],
            'temperature':
            0,
            'max_tokens':
            10
        }

        response = requests.Session().post(
            "https://api.endpoints.anyscale.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer " + os.environ["ANYSCALE_API_KEY"]
            },
            json=data)

        completion = response.json().get('choices',
                                         [{}])[0].get('message',
                                                      {}).get('content', '')
        completion = completion.split('#')[0].split('\n')[0].split(
            'Previously')[0].split('/')[0].split(';')[0]
        completion = completion.lstrip()  # Remove leading white space

        completions.append(completion)  # Append the completion to the list

    return render(
        request,
        'show_completion.html',
        {
            'completions': completions,  # Return the list of completions
            'issue': issue
        })


@csrf_exempt
@require_http_methods(["GET"])
def fetch_openai_completion(
    request,
    issue,
    prompt="Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######Previously Mentioned Issues () "
):
    issues_list = issue.split(';')  # Split the issue string into a list
    completions = []  # List to hold completions for each issue

    for single_issue in issues_list:
        try:
            previous_list = ', '.join(find_similar_issues(single_issue))
        except:
            previous_list = ''

        data = {
            'model':
            'gpt-3.5-turbo',
            'messages': [{
                'role':
                'system',
                'content':
                f"Previously Mentioned Issues ({previous_list}) {prompt}{single_issue}->"
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
        completions.append(completion)  # Append the completion to the list

    return render(
        request,
        'show_completion.html',
        {
            'completions': completions,  # Return the list of completions
            'issue': issue
        })


@csrf_exempt
@require_http_methods(["GET", "POST"])
def dynamic_issue_openai(request):

    # Check if the request is POST and parse JSON data
    if request.method == 'POST':
        data = json.loads(request.body)
        issue = data.get('input')
    else:  # For GET request
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
        'gpt-3.5-turbo',
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

    # Get midpoint
    midpoint = (float(get_global_setting('min_scale')) +
                float(get_global_setting('max_scale'))) / 2

    # If midpoint is empty, set it to 3
    if not midpoint:
        midpoint = 3

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
            avg_rating=midpoint,
            ratings=1,
            var_rating=1,
            embedding=embed_completion)
        dynamic_issue_question.save()  # Save the instance to the database

    return JsonResponse({'completion': completion})


@csrf_exempt
@require_http_methods(["GET", "POST"])
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

    completion = completion.strip().split('#')[0].strip().split(
        '\n')[0].strip().split('Previously')[0].strip().split(
            '/')[0].strip().split(';')[0].strip()
    embed_completion = generate_embedding_alt(completion)

    # Get midpoint
    midpoint = (float(get_global_setting('min_scale')) +
                float(get_global_setting('max_scale'))) / 2

    # If midpoint is empty, set it to 3
    if not midpoint:
        midpoint = 3

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
            avg_rating=midpoint,
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


@csrf_exempt
@require_http_methods(["POST"])
def update_ratings(request):
    logger = logging.getLogger(__name__)

    try:
        data = json.loads(request.body)
        logger.info("Received data: %s", data)

        ratings = data.get('ratings', [])
        if not ratings:
            logger.error("No ratings provided in the request.")
            return JsonResponse(
                {
                    'status': 'error',
                    'message': 'No ratings provided.'
                },
                status=400)

        for rating_info in ratings:
            question_text = rating_info.get('question')
            user_id = rating_info.get('user_id')
            rating = float(rating_info.get('rating'))
            logger.info("Processing rating for question '%s' with rating %f",
                        question_text, rating)

            # Check if session min_scale and max_scale are set. If not, set to 1 and 5 respectively
            min_scale = float(get_global_setting('min_scale'))
            max_scale = float(get_global_setting('max_scale'))

            if rating < min_scale or rating > max_scale:
                continue  # Skip invalid ratings

            # Retrieve the question from the database and update the ratings
            try:
                question = DynamicIssueQuestion.objects.get(
                    question=question_text)
                question.avg_rating = (
                    (question.avg_rating * question.ratings) +
                    rating) / (question.ratings + 1)
                question.ratings += 1
                question.var_rating = 1 / question.ratings
                question.save()

                UserDatabase.objects.update_or_create(
                    user_id=user_id,
                    question=question_text,
                    defaults={'rating': rating})

            except DynamicIssueQuestion.DoesNotExist:
                # Skip if question not found
                continue

        return JsonResponse({
            'status': 'success',
            'message': 'Ratings updated successfully.'
        })

    except json.JSONDecodeError as e:
        logger.exception("JSON decode error: %s", e)
        return JsonResponse(
            {
                'status': 'error',
                'message': 'Invalid JSON format.'
            }, status=400)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    logger.info("Ratings updated successfully.")
    return JsonResponse({
        'status': 'success',
        'message': 'Ratings updated successfully.'
    })


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

    # Run 5000 simulations
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

    # Get midpoint
    midpoint = (float(get_global_setting('min_scale')) +
                float(get_global_setting('max_scale'))) / 2
    # If midpoint is empty, set it to 3
    if not midpoint:
        midpoint = 3

    num_items_str = get_global_setting('num_items')
    print(num_items_str)
    # Set a default value if num_items is not found or if it's None
    num_items = int(num_items_str) if num_items_str is not None else midpoint
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


def survey_view(request):
    min_scale = get_global_setting('min_scale')  # Default to 1 if not set
    max_scale = get_global_setting('max_scale')  # Default to 5 if not set
    survey_text = get_global_setting('survey_text')

    if survey_text is None:
        survey_text = 'Rate the questions in importance on a 1-5 scale.'

    # Add them to the context
    context = {
        'min_scale': min_scale,
        'max_scale': max_scale,
        'survey_text': survey_text
    }

    return render(request, 'survey.html', context)
