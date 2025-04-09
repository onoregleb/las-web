import sys, os
sys.path.append('/путь/до/сайта')
sys.path.append('/путь/до/папки/с/библиотеками')
from a2wsgi import ASGIMiddleware
from main import app
application = ASGIMiddleware(app)