
import cv2
from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse

# Create your views here.
from django.shortcuts import resolve_url
from django.views.decorators.csrf import csrf_exempt

from api.utils import FaceMarker

base_dir = '/tmp/landmark'
face_marker = FaceMarker(base_dir)

@csrf_exempt
def detect_landmark(request):
    if request.method == 'POST':
        filename = face_marker.save_temp(request.FILES.get('image'))
        if not filename:
            return HttpResponse('invalid file', status=500)

        frame = cv2.imread(filename)

        try:
            face_marker.detect_face(frame)
        except Exception:
            return HttpResponse('invalid file', status=500)

        tokens = filename.split('/')
        filename = tokens[len(tokens) - 1]
        filepath = f'{settings.MEDIA_ROOT}/{filename}'
        if face_marker.save_image_from_cv2(frame, filepath):
            return JsonResponse({'img_url': f'{settings.MEDIA_URL}{filename}'})
        else:
            return HttpResponse('failed detect face', status=500)
    else:
        return HttpResponse()
