from django.shortcuts import render,redirect
import cv2
from demo import give_me_out_res as produce
import os
from PIL import Image
from io import BytesIO


from complaint.models import Complaint

# Create your views here.

def complaint(request):
    if (request.user.is_authenticated):
        return render(request,'complaint/complaint.html')
    else:
        return render(request,'user/signin.html')

def add_complaint(request):
    if request.method == "POST":
        user = request.user
        violation_date=request.POST.get("vdate")
        place = request.POST.get("place")
        district=request.POST.get("district")
        state=request.POST.get("state")
        comment=request.POST.get("comment")
        uploaded_file = request.FILES.get('evidence')
        complaint = Complaint(
                    user=user,
                    violation_date=violation_date,
                    place=place,
                    district=district,
                    state=state,
                    comment=comment,
                    # Set other fields similarly
                    uploaded_file=uploaded_file,
                )

        # Save the model to the database
        complaint.save()

        verify(request, complaint.id)
        content={
            "success":1
        }
        return render(request,'pages/index.html',content)

# views.py
# from django.shortcuts import render, redirect
# from .forms import ComplaintForm

# def create_complaint(request):
#     if request.method == 'POST':
#         form = ComplaintForm(request.POST, request.FILES)
#         if form.is_valid():
#             complaint = form.save(commit=False)

#             # Save file to the model instance
#             complaint.uploaded_file = form.cleaned_data['uploaded_file']

#             # Save the model to the database
#             complaint.save()

#             # Do any additional processing or redirect as needed
#             return redirect('success_page')
#     else:
#         form = ComplaintForm()

#     return render(request, 'complaint/complaint.html', {'form': form})

def complaints(request):
    complaints=Complaint.objects.filter(user=request.user).order_by('-id')
    content={
        "complaints":complaints
    }
    return render(request,'complaint/complaints.html',content)

def cancel(request,id):
    Complaint.objects.filter(id=id).delete()
    content={
            "delete":1
        }
    return render(request,'pages/index.html',content)


# ML part


def verify(request, cid):
    complaint=Complaint.objects.get(id=cid)
    print(complaint)
    print(complaint.uploaded_file.path)
    # vdo = cv2.VideoCapture(complaint.uploaded_file.path)
    
    res,mybikeNumber,cropped=produce(complaint.uploaded_file.path)
    # image_path = r'./'
    # with open(image_path, 'rb') as img_file:
    # complaint.evidence_photo.save('cropped_image.jpg', img_file)
    # print(1)
    cv2.imwrite(f'media/complaint_photos/cropped_image{cid}.jpg', cropped)
    

    if res==1:
        complaint.status='Processed'
        complaint.bike_number=mybikeNumber
        complaint.evidence_photo = f'complaint_photos/cropped_image{cid}.jpg'
        complaint.save()
    elif res==2:
        complaint.status='Initiated'
        complaint.bike_number=mybikeNumber
        complaint.evidence_photo = f'complaint_photos/cropped_image{cid}.jpg'

        complaint.save()
    else:
        complaint.status='Rejected'
        complaint.save()
    print(complaint)