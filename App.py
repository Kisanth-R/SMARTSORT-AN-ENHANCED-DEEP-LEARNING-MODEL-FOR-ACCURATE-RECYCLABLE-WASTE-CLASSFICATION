from flask import Flask, render_template, flash, request, session

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Prediction")
def Prediction():
    return render_template('Prediction.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        import tensorflow as tf
        import numpy as np
        import cv2
        from keras.preprocessing import image
        file = request.files['file']
        file.save('static/upload/Test.jpg')
        org = 'static/upload/Test.jpg'

        img1 = cv2.imread('static/upload/Test.jpg')

        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        noi = 'static/upload/noi.jpg'
        cv2.imwrite(noi, dst)

        classifierLoad = tf.keras.models.load_model('Vggmodel.h5')
        test_image = image.load_img('static/upload/Test.jpg', target_size=(100, 100))
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        print(result)

        result = classifierLoad.predict(test_image)
        print(result)
        result = np.argmax(result, axis=1)

        print(result)

        out = ''
        pre = ''
        if result[0] == 0:
            print("cardboard")
            out = "cardboard"
            pre = "Degradable"
        elif result[0] == 1:
            print("glass")
            out = "glass"
            pre = "Non-Degradable"
        elif result[0] == 2:
            print("metal")
            out = "metal"
            pre = "Non-Degradable"
        elif result[0] == 3:
            print("paper")
            out = "paper"
            pre = "Degradable"

        elif result[0] == 4:
            print("plastic")
            out = "plastic"
            pre = "Non-Degradable"
        elif result[0] == 5:
            print("trash")
            out = "trash"
            pre = "Degradable"

        sendmsg("9384525930", 'Prediction Result : ' + out + ' ' + pre)

        return render_template('Result.html', res=out, pre=pre, org=org, noi=noi)


def sendmsg(targetno, message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + str(
            targetno) + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")


@app.route("/Camera")
def Camera():
    import cv2
    from ultralytics import YOLO

    dd1 = 0

    # Load the YOLOv8 model
    model = YOLO('runs/detect/train7/weights/best.pt')
    # Open the video file
    # video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(0)
    res = ''

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.7)
            for result in results:
                if result.boxes:
                    box = result.boxes[0]
                    class_id = int(box.cls)
                    object_name = model.names[class_id]
                    print(object_name)

                    if object_name != '':
                        dd1 += 1

                    if dd1 == 50:
                        dd1 = 0

                        annotated_frame = results[0].plot()

                        if object_name == 'BIODEGRADABLE':
                            res = "BIODEGRADABLE"
                        elif object_name == 'CARDBOARD':
                            res = ("BIODEGRADABLE"),
                        elif object_name == 'GLASS':
                            res = ("NON-DEGRADABLE"),
                        elif object_name == 'METAL':
                            res = ("NON-DEGRADABLE"),
                        elif object_name == 'PAPER':
                            res = ("BIODEGRADABLE"),
                        elif object_name == 'PLASTIC':
                            res = ("NON-DEGRADABLE"),

                        sendmsg("9384525930", 'Prediction Result : ' + str(object_name) + ' ' + str(res))
                        # Optionally, visualize the results
                        annotated_frame = results[0].plot()
                        outi = "static/Out/out.jpg"
                        cv2.imwrite("static/Out/out.jpg", annotated_frame)

                        sendmail()

                        import winsound

                        filename = 'alert.wav'
                        winsound.PlaySound(filename, winsound.SND_FILENAME)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')


def sendmail():
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "projectmailm@gmail.com"
    toaddr = "yellameydooop@gmail.com"

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = "Waste Classification "

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # open the file to be sent
    filename = "alert.jpg"
    attachment = open("static/Out/out.jpg", "rb")

    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    p.set_payload((attachment).read())

    # encode into base64
    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "qmgn xecl bkqv musr")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()


if __name__ == '__main__':
    # app.run(host='0.0.0.0',debug = True, port = 5000)
    app.run(debug=True, use_reloader=True)
