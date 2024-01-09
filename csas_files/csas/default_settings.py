from .models import GlobalSetting


def load_default_settings():
    # Define your default settings
    default_settings = {
        'min_scale': '1',
        'max_scale': '5',
        'num_items': '3',
        'ai_choice': 'openai',
        'prompt':
        'You are a classification expert who takes an input and returns a political issue as a summary. Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######',
        'survey_text': 'Rate the importance of this issue on a 1-5 scale.'
    }

    for key, value in default_settings.items():
        GlobalSetting.objects.update_or_create(key=key,
                                               defaults={'value': value})
