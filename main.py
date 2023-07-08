# Ozner Leyva Mariscal A01742377
# Carolina González Leal A01284948
# Erick Siller Ojeda A01382929
# Valeria Enríquez Limón A00832782
# Santiago Martínez Vallejo A00571878

import numpy as np

# Ruta del archivo de texto
# archivo = print(input("Ingresa la matriz: "))
archivo = '01. Matrix_A_16_2_4.txt'

# Leer los datos del archivo
datos = np.genfromtxt(archivo, delimiter=',')

# Redimensionar la matriz
filas, columnas = datos.shape
matriz = datos.reshape((16, 16))
matriz = matriz.astype(int)

# Imprimir la matriz
# print(matriz)


# Strassen's Algorithm
def strassen(a, b):
  global acum  # Counter variable

  n = a.shape[0]

  # Base case for recursion
  if n == 1:
    acum += 1
    return a * b

  # Get the dividing len
  mid = n // 2

  # Split the a matrix
  a11 = a[:mid, :mid]
  a12 = a[:mid, mid:]
  a21 = a[mid:, :mid]
  a22 = a[mid:, mid:]

  # Split the b matrix
  b11 = b[:mid, :mid]
  b12 = b[:mid, mid:]
  b21 = b[mid:, :mid]
  b22 = b[mid:, mid:]

  # Recursive steps
  p1 = strassen(a11 + a22, b11 + b22)
  p2 = strassen(a21 + a22, b11)
  p3 = strassen(a11, b12 - b22)
  p4 = strassen(a22, b21 - b11)
  p5 = strassen(a11 + a12, b22)
  p6 = strassen(a21 - a11, b11 + b12)
  p7 = strassen(a12 - a22, b21 + b22)

  # Compute the quadrants of the resulting matrix
  c11 = p1 + p4 - p5 + p7
  c12 = p3 + p5
  c21 = p2 + p4
  c22 = p1 - p2 + p3 + p6

  # Combine the quadrants into a single matrix
  c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

  return c


# Textbook Definition Algorithm
def textbook_matrix_multiply(A, B):
  if len(A[0]) != len(B):
    raise ValueError(
      "Number of columns in A must be equal to number of rows in B")

  # Dimensions of the resulting matrix
  rows_A = len(A)
  cols_B = len(B[0])

  # Create a new matrix to store the result
  C = [[0] * cols_B for _ in range(rows_A)]

  # Counter for number of multiplicactions
  num_multiplications = 0

  # Iterate over rows of A
  for i in range(rows_A):
    # Iterate over columns of B
    for j in range(cols_B):
      # Iterate over columns of A (or rows of B)
      for k in range(len(B)):
        C[i][j] += A[i][k] * B[k][j]
        num_multiplications += 1

  return C, num_multiplications


# In this program, we have used nested for loops to iterate through each row and each column. We accumulate the sum of products in the result.
# This technique is simple but computationally expensive as we increase the order of the matrix.

acum = 0  # acumulador de strassen


# Menu para elegir función y leer archivos txt de entrada
def menu():
  while True:
    print("Matrix Multiplication Menu: ")
    print("1-Strassen Method")
    print("2-Textbook Definition Algorithm")
    print("0-Salir del programa")
    res = int(input("Introduzca la opción: "))

    if res == 1:
      m_A = input("Enter the filename for matrix A: ")
      m_B = input("Enter the filename for matrix B: ")

      matrix_data_A = np.loadtxt(m_A, delimiter=',')
      matrix_data_B = np.loadtxt(m_B, delimiter=',')

      matrix_array_A = np.array(matrix_data_A)
      matrix_array_B = np.array(matrix_data_B)

      result = strassen(matrix_array_A, matrix_array_B)
      print("The result of the multiplication is: ", result)
      print("The number of multiplications were: ", acum)

    elif res == 2:
      m_A = input("Enter the filename for matrix A: ")
      m_B = input("Enter the filename for matrix B: ")

      matrix_data_A = np.loadtxt(m_A, delimiter=',')
      matrix_data_B = np.loadtxt(m_B, delimiter=',')

      result, mult_num = textbook_matrix_multiply(matrix_data_A, matrix_data_B)
      print("The result of the multiplication is: ", result)
      print("The number of multiplications were: ", mult_num)

    elif res == 3:
      break

    else:
      print("Opción inválida")


menu()

# Example usage
#a = np.array([[2, 0, 1, 2], [3, 0, 0, 2], [5, 1, 1, 2], [1, 2, 3, 2]])
#b = np.array([[1, 0, 1, 2], [1, 2, 1, 2], [1, 1, 0, 2], [1, 2, 3, 2]])
# en este ejemplo con strassen es 49, normal es 64

#result = strassen(a, b)
#print(result)
#print("Multiplications:", acum)
# print(acum)
