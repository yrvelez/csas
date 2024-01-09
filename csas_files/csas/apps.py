from django.apps import AppConfig


class CsasConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'csas'

    def ready(self):
        from . import default_settings  # Import your module
        default_settings.load_default_settings()
