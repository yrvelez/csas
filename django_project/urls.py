from django.contrib import admin
from django.urls import path  # This import statement is necessary
from csas.views import dynamic_issue_openai, dynamic_issue_mistral, view_issues, update_rating, select_questions_simulation, main_page, fetch_openai_completion, upload_completion, view_db_content, fetch_mistral_completion, save_session_data, upload_completions, survey_view, update_ratings, view_user_content

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dynamic-issue-oai/',
         dynamic_issue_openai,
         name='dynamic_issue_openai'),
    path('dynamic-issue-mistral/',
         dynamic_issue_mistral,
         name='dynamic_issue_mistral'),
    path('save_session/', save_session_data, name='save_session'),
    path('view-issues/', view_issues, name='view_issues'),
    path('update-rating/', update_rating, name='update_rating'),
    path('select-questions-simulation/',
         select_questions_simulation,
         name='select_questions_simulation'),
    path('', main_page, name='main_page'),
    path('fetch-oai-completion/<str:issue>/<str:prompt>',
         fetch_openai_completion,
         name='fetch_openai_completion'),
    path('fetch-mistral-completion/<str:issue>/<str:prompt>',
         fetch_mistral_completion,
         name='fetch_mistral_completion'),
    path('upload-completion/', upload_completion, name='upload_completion'),
    path('upload-completions/', upload_completions, name='upload_completions'),
    path('view-db-content/', view_db_content, name='view_db_content'),
    path('view-user-content/', view_user_content, name='view_user_content'),
    path('survey/', survey_view, name='survey'),
    path('update-ratings/', update_ratings, name='update_ratings')
]
