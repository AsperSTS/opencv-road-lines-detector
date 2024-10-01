import cv2
import numpy as np

def fill_polygon(mask, points, color=255):
    """
    Funcion para rellenar la mascara, marcar con un valor de 255 los pixeles que nos interesan con lineas
    """
    height, width = mask.shape[:2]  # Obtenemos el alto y ancho de la mascara

    points = np.array(points, dtype=np.int32)
    
    # Encontramos el límite de la caja que contiene el polígono
    x_min = max(0, np.min(points[:, 0]))
    x_max = min(width - 1, np.max(points[:, 0]))
    y_min = max(0, np.min(points[:, 1])) # Obtenemos el valor "y" Minimo para pintar el poligono
    y_max = min(height - 1, np.max(points[:, 1]))   # Obtenemos el valor maximo de "y" en los puntos
    
    # Iteramos en el rango que encontramos anteriormente para pintar el polígono
    for y in range(y_min, y_max + 1):
        # Encontrar los segmentos donde el polígono cruza una línea horizontal
        intersections = []
        # Iteramos la cantidad de puntos que hayamos definido en nuestro array
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]  # Conecta el último punto con el primero cuando n/k = 0

            # Verificamos si los puntos forman una linea horizontal 
            if (y1 < y and y2 >= y) or (y2 < y and y1 >= y):
                # Encontrar la intersección con la línea horizontal
                x_intersection = x1 + (y - y1) * (x2 - x1) // (y2 - y1)
                # Agregamos la interseccion a la lista de intersecciones
                intersections.append(x_intersection)
        
        # Ordenamos las intersecciones de forma ascendente
        intersections.sort()
        
        # Dibujamos las lineas entre dos puntos de interseccion
        for i in range(0, len(intersections), 2):
            x_start = max(0, intersections[i])
            x_end = min(width - 1, intersections[i + 1])
            if x_start <= x_end:  # Verificamos que x_start no sea mayor que x_end
                mask[y, x_start:x_end + 1] = color  # Rellenar desde x_start hasta x_end con puntos 

def equalize_histogram(v_channel):
    # Paso 1: Calculamos el histograma de la imagen
    hist, _ = np.histogram(v_channel.flatten(), 256, [0, 256])

    # Paso 2: Calculamos la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()
    
    # Normalizamos el CDF 
    cdf_normalized = cdf * (255.0 / cdf[-1])

    # Primero, buscamos que los valores sean enteros
    cdf_normalized = np.ma.masked_equal(cdf_normalized, 0)  # Ocultar valores 0
    cdf_normalized = (cdf_normalized - cdf_normalized.min()) * 255 / (cdf_normalized.max() - cdf_normalized.min())
    cdf_normalized = np.ma.filled(cdf_normalized, 0).astype('uint8')  # Rellenamod y convertimos a uint8

    # Paso 4: Aplicamos el remapeo a la imagen original usando la CDF
    equalized_v = cdf_normalized[v_channel]

    return equalized_v

def autocontrast(v_channel, alpha=0, beta=255):
    # Paso 1: Buscamos el valor mínimo y máximo del canal V
    v_min = np.min(v_channel)
    v_max = np.max(v_channel)
    
    # Eliminamos los valores que nos den 0
    if v_max - v_min == 0:
        return np.full(v_channel.shape, alpha, dtype=np.uint8)
    
    # Paso 2: Aplicamos la fomula del autocontraste
    v_normalized = ((v_channel - v_min) / (v_max - v_min)) * (beta - alpha) + alpha
    
    # Paso 3: Convertimos a uint8 
    v_normalized = np.clip(v_normalized, alpha, beta).astype(np.uint8)
    
    """ Documentacion de np.clip
    np.clip : Given an interval, values outside the interval are clipped to the interval edges. 
    For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, 
    and values larger than 1 become 1.
    """
    
    return v_normalized

def restricted_autocontrast(v_channel, lower_percentile=2, upper_percentile=98, alpha=0, beta=255):
    """
    Aplica autocontraste restringido al canal V de la imagen.
    
    :param v_channel: Canal V de la imagen (numpy array).
    :param lower_percentile: Percentil inferior para el ajuste de contraste.
    :param upper_percentile: Percentil superior para el ajuste de contraste.
    :param alpha: Valor mínimo para la salida.
    :param beta: Valor máximo para la salida.
    :return: Canal V con autocontraste restringido.
    """
    
    # Paso 1: Calcula los percentiles
    lower_bound = np.percentile(v_channel, lower_percentile)
    upper_bound = np.percentile(v_channel, upper_percentile)
    
    # Paso 2: Verifica si el rango es válido
    if upper_bound - lower_bound == 0:
        return np.full(v_channel.shape, alpha, dtype=np.uint8)
    
    # Paso 3: Aplica la fórmula de autocontraste restringido
    v_normalized = ((v_channel - lower_bound) / (upper_bound - lower_bound)) * (beta - alpha) + alpha
    
    # Paso 4: Convierte a uint8
    v_normalized = np.clip(v_normalized, alpha, beta).astype(np.uint8)
    
    return v_normalized 

def _gamma(v, gamma_val):
    # Paso 1: Dividimos cada valor de nuestro canal v por el valor de quantizacion
    normalized_v = v / 255.0
    # Paso 2: Elevamos al valor de gama
    corrected_v = np.power(normalized_v, gamma_val)
    # Paso 3: Multiplicamos por 255 para volver al rango de 0 a 255
    corrected_v = np.uint8(corrected_v * 255)    
    # Retornamos el valor con el operador gamma aplicado
    return corrected_v

def create_color_mask(lower_white, upper_white, lower_yellow, upper_yellow, hsv_corrected):
    # Creamos las máscaras usando operaciones numpy
    mask_yellow = np.all((hsv_corrected >= lower_yellow) & (hsv_corrected <= upper_yellow), axis=-1).astype(np.uint8) * 255
    mask_white = np.all((hsv_corrected >= lower_white) & (hsv_corrected <= upper_white), axis=-1).astype(np.uint8) * 255
    
    # Aclaramos el canal de valor (V) en las áreas donde las máscaras sean 255 (colores detectados)
    hsv_corrected[:, :, 2] = np.where(mask_yellow == 255, np.minimum(hsv_corrected[:, :, 2] + 255, 255), hsv_corrected[:, :, 2])
    hsv_corrected[:, :, 2] = np.where(mask_white == 255, np.minimum(hsv_corrected[:, :, 2] + 255, 255), hsv_corrected[:, :, 2])
    
    return mask_yellow, mask_white

def binarize_frame(image, threshold_value, max_value):
    # Usar una operación vectorizada para optimizar la implementacion
    binary_image = np.where(image > threshold_value, max_value, 0).astype(np.uint8)
    return binary_image
   
def main():
    # Abrimos el video 
    video = cv2.VideoCapture('lineas.mp4')

    # Obtenemos los fps del video para poder calcular la pausa del video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Verificamos si el video se abrio correctamente
    if not video.isOpened():
        print("Error al abrir el video.")
        exit()
    # Obtenemos el primer frame de video para poder calcular el ancho y largo del mismo
    ret, frame = video.read()
    # Verificamos si pudimos obtener el primer frame 
    if not ret:
        print("No se pudo obtener el primer frame.")
        video.release()
        exit()    

    # Dimensiones del frame
    height, width = frame.shape[:2]

    # Parámetros para ajustar las líneas de fuga
    left_shift = 0     #  desviación izquierda de la línea de fuga
    right_shift = 0    # desviación derecha de la línea de fuga
    top_shift = -70      # Ajusta qué tan arriba empiezan las líneas de fuga
    bottom_shift = 220  # Ajusta qué tan abajo empiezan las líneas de fuga


    # Definimos los rangos para el color amarillo
    lower_yellow = np.array([15, 100, 100])  
    upper_yellow = np.array([28, 255, 255])

    # Definimos los rangos para el color blanco
    lower_white = np.array([0, 0, 200])  
    upper_white = np.array([180, 20, 255])  
    
    # Creamos una máscara negra con las dimensiones del frame
    vanishing_lines_mask = np.zeros((height, width), dtype=np.uint8)

    # Definimos el poligono de nuestras las líneas de fuga 
    polygon_points = np.array([
        (left_shift, height - bottom_shift),
        (left_shift, height),
        (width - right_shift, height),
        (width - right_shift, height - bottom_shift),
        ((width // 12) * 7, height // 2 - top_shift),
        ((width // 12) * 5, height // 2 - top_shift)

    ], dtype=np.int32)

    # Dibujar el área de las líneas de fuga en la máscara
    fill_polygon(vanishing_lines_mask, polygon_points)
    
    # Variables normalizadas
    gamma = 3  # Valor inicial de gamma
    is_paused = False
    previous_hist = None  # Histograma anterior
    change_threshold = 0.05  # Umbral de cambio
    tmp = 0

    while True:
        if not is_paused:
            ret, frame = video.read()

            # Verificar si se leyó correctamente
            if not ret:
                break
            # Convertimos el cuadro a otro espacio de color
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # # Calculamos el histograma del canal v donde vamos a trabajar
            # hist = cv2.calcHist([hsv_frame[...,2]], [0], None, [256], [0, 256]).flatten()
            # # Comprobamos si existe histograma anterior
            # if previous_hist is not None:
            #     # Normalizamos la diferencia entre el histograma actual y el anterior
            #     hist_diff = np.abs(hist - previous_hist) / (np.sum(hist) + 1e-6)  
            #     # Calculamos el cambio total de la diferencia entre histogramas
            #     total_change = np.sum(hist_diff)
            #     # Calculamos el cambio en la variable de cambio
            #     if total_change < change_threshold:
            #         tmp -= 1
            #     elif total_change > change_threshold:
            #         tmp += 1
            #     tmp = max(0, min(tmp, 30)) 

            #     print(f"Changes: {total_change}, Var: {tmp}")
            # # Actualizamos el hisograma previo con el actual        
            # previous_hist = hist
            
            # Normalizar el canal V antes de aplicar gamma
            v_autocontrast = restricted_autocontrast(hsv_frame[...,2])
            # Aplicamos ecualizacion del histograma pasandole el frame normalizado
            v_equalized = equalize_histogram(v_autocontrast)
            # Aplicamos el operador gamma 

            
            v_gamma_corrected = _gamma(v_equalized, gamma)
            # Actualizar directamente el canal V en hsv_frame
            hsv_frame[..., 2] = v_gamma_corrected
            # Obtenemos el frame en BGR para testeos
            bgr_result = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
            # Creamos las mascaras de color con los rangos definidosanteriormente
            mask_yellow, mask_white = create_color_mask(lower_white, upper_white, lower_yellow, upper_yellow, hsv_frame) 
            
            # frame_enhanced = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR) # Variables de prueba
                        
            # Aplicar la máscara de color a la imagen original
            color_mask = cv2.bitwise_or(mask_yellow, mask_white)
            # Obtenemos el frame con los colores de las mascaras resaltados 
            two_mask_result = cv2.bitwise_and(bgr_result, bgr_result, mask=color_mask) 
            # Convertimos el frame a escala de grises para obtener la máscara de líneas de fuga
            result_gray = cv2.cvtColor(two_mask_result, cv2.COLOR_BGR2GRAY)
            
            # binarization_threshold = 50 + tmp  # Ajusta el umbral base en función de los cambios
            # Binarizamos el resultado~
            frame_binary = binarize_frame(result_gray, 0, 255)
            # frame_binary = binarize_frame(result_gray, binarization_threshold, 255)
            
            # Aplicamos la mascara que contiene nuestras lineas de fuga al frame
            masked_frame = cv2.bitwise_and(frame_binary, frame_binary, mask=vanishing_lines_mask)
            # Mostramos el frame 
            cv2.imshow('Processed Video', masked_frame)
            # cv2.imshow('Normalized Video', frame_enhanced)
        
        # Capturar eventos de teclado
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        
        if key == ord('q'):  # Salir al presionar 'q'
            break
        elif key == ord(' '):  # Pausar/Reanudar al presionar la barra espaciadora
            is_paused = not is_paused

    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()