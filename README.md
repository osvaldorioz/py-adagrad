### **Resumen del Algoritmo Adagrad**  
Adagrad (Adaptive Gradient Algorithm) es un optimizador que ajusta dinámicamente la tasa de aprendizaje para cada parámetro en función del historial acumulado de gradientes. La actualización de cada peso \( w_i \) sigue la fórmula:

![imagen](https://github.com/user-attachments/assets/ffd2ac6f-e1fb-480a-bf2f-538228deb197)


### **Implementación en el Programa**  
1. **En C++ (Pybind11)**  
   - Se creó una clase `Adagrad` que mantiene los pesos y la acumulación de gradientes.  
   - En cada iteración, la función `update` actualiza los pesos usando la fórmula de Adagrad.  
   - Se expone la clase a Python con Pybind11.

2. **En Python**  
   - Se generan datos de prueba para una regresión lineal.  
   - Se inicializa `Adagrad` y se ejecutan varias iteraciones para optimizar los parámetros.  
   - Se grafican los datos ajustados y la evolución de la pérdida.  

Esta implementación demuestra cómo Adagrad adapta la tasa de aprendizaje automáticamente, mejorando la convergencia en problemas de optimización.
