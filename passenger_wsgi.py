import sys, os
sys.path.append('/home/x/x96132vo/las-analyze/')
sys.path.append('/home/x/x96132vo/.local/lib/python3.10/site-packages')
from a2wsgi import ASGIMiddleware
from main import app
application = ASGIMiddleware(app)