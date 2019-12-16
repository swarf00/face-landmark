
import cv2
from django.http import FileResponse, HttpResponse


# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from api.utils import FaceMarker

face_marker = FaceMarker()

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

        if face_marker.save_image_from_cv2(frame, filename):
            return FileResponse(open(filename, 'rb'))
        else:
            return HttpResponse('failed detect face', status=500)
    else:
        return HttpResponse()


