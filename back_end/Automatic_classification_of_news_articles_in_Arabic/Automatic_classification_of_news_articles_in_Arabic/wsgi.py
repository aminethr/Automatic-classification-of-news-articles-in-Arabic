"""
WSGI config for Automatic_classification_of_news_articles_in_Arabic project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "Automatic_classification_of_news_articles_in_Arabic.settings",
)

application = get_wsgi_application()
