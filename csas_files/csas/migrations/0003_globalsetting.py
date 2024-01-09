# Generated by Django 3.2.23 on 2024-01-08 21:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('csas', '0002_userdatabase'),
    ]

    operations = [
        migrations.CreateModel(
            name='GlobalSetting',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('key', models.CharField(max_length=255, unique=True)),
                ('value', models.TextField()),
            ],
        ),
    ]