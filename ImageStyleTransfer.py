# Importamos las bibliotecas necesarias para el transfer de estilo
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargamos la imagen que queremos aplicar el transfer de estilo
imagen = tf.keras.preprocessing.image.load_img("ruta/de/la/imagen.jpg")

# Definimos el estilo que queremos aplicar a la imagen
estilo = tf.keras.preprocessing.image.load_img("ruta/del/estilo.jpg")

# Creamos un modelo de transfer de estilo usando el estilo definido
modelo = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=imagen.size)

# Generamos una representación en tensor de la imagen y del estilo
imagen_tensor = tf.keras.applications.vgg19.preprocess_input(tf.keras.preprocessing.image.img_to_array(imagen))
estilo_tensor = tf.keras.applications.vgg19.preprocess_input(tf.keras.preprocessing.image.img_to_array(estilo))

# Aplicamos el transfer de estilo a la imagen
imagen_procesada = modelo.predict(imagen_tensor)

# Mostramos la imagen original y la imagen con el transfer de estilo aplicado
plt.figure()
plt.imshow(imagen)
plt.figure()
plt.imshow(imagen_procesada)
plt.show()


### VIIIIDEOOOOOO



# Importamos las bibliotecas necesarias para el transfer de estilo y la captura de video
import tensorflow as tf
import cv2

# Creamos una instancia de la cámara
captura = cv2.VideoCapture(0)

# Definimos el estilo que queremos aplicar a la imagen
estilo = tf.keras.preprocessing.image.load_img("ruta/del/estilo.jpg")

# Creamos un modelo de transfer de estilo usando el estilo definido
modelo = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=captura.size)

# Creamos un bucle para capturar y procesar cada fotograma del video
while(True):
    # Leemos un fotograma del video
    ret, frame = captura.read()

    # Si no hay más fotogramas, terminamos el bucle
    if not ret:
        break

    # Generamos una representación en tensor del fotograma
    frame_tensor = tf.keras.applications.vgg19.preprocess_input(tf.keras.preprocessing.image.img_to_array(frame))

    # Aplicamos el transfer de estilo al fotograma
    frame_procesado = modelo.predict(frame_tensor)

    # Mostramos el fotograma original y el fotograma con el transfer de estilo aplicado
    cv2.imshow("Original", frame)
    cv2.imshow("Transfer de estilo", frame_procesado)

    # Si se pulsa la tecla 'q', terminamos el bucle
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberamos la cámara y cerramos las ventanas de video
captura.release()
cv2.destroyAllWindows()