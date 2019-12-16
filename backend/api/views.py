from django.http import FileResponse, HttpResponse


# Create your views here.
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def detect_landmark(request):
    if request.method == 'POST':
        print(request.FILES)
        return FileResponse(request.FILES)
    else:
        return HttpResponse()