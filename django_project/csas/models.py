from django.db import models

class DynamicIssueQuestion(models.Model):
    question = models.TextField()
    avg_rating = models.FloatField(default=3.0)
    ratings = models.IntegerField(default=1)
    var_rating = models.FloatField(default=1.0)
    embedding = models.JSONField(default=None)

    def __str__(self):
        return self.question

    class Meta:
        app_label = 'csas'
