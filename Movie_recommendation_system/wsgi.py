import os
from django.core.wsgi import get_wsgi_application
import django
from django.core.management import call_command

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Movie_recommendation_system.settings')

django.setup()
call_command('migrate')  # 自动执行migrate

application = get_wsgi_application()
