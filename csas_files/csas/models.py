from django.db import models


class DynamicIssueQuestion(models.Model):
    question = models.TextField()
    avg_rating = models.FloatField(default=3.0)
    ratings = models.IntegerField(default=1)
    var_rating = models.FloatField()
    embedding = models.JSONField(default=None)

    def __str__(self):
        return self.question

    class Meta:
        app_label = 'csas'


class UserDatabase(models.Model):
    user_id = models.TextField()
    question = models.TextField()
    rating = models.FloatField(default=3.0)

    def __str__(self):
        return self.id

    class Meta:
        app_label = 'csas'


class GlobalSetting(models.Model):
    key = models.CharField(max_length=255, unique=True)
    value = models.TextField()

    def __str__(self):
        return f"{self.key}: {self.value}"
