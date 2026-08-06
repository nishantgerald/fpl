# --timeout 120: a screenshot import is a synchronous model call that runs
# 10-40s once the image has crossed the relay. Gunicorn's default is 30s,
# after which it kills the worker mid-request and the user gets a 500 with
# nothing in it — which is how a working import looked like a broken one.
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
