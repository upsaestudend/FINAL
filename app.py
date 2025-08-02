
import streamlit as st
import numpy as np
import joblib

# Cargar el modelo
modelo = joblib.load("modelo_entrenado.pkl")

# Función para calcular bono, TP modificada y Final modificada
def calcular_notas(tp, asistencia, final):
    bono = 0
    if asistencia > 95:
        bono = 0.20 * tp
    tp_modificada = tp + bono
    final_modificada = final if asistencia >= 80 else 0
    return bono, tp_modificada, final_modificada

# Función para clasificar nota final
def clasificar(nota):
    if nota >= 91:
        return "Excelente"
    elif nota >= 81:
        return "Óptimo"
    elif nota >= 71:
        return "Satisfactorio"
    elif nota >= 61:
        return "Bueno"
    elif nota >= 51:
        return "Regular"
    else:
        return "Insuficiente"

# Título
st.title("Predicción de Nota Final de Estudiantes")

# Entradas del usuario
st.subheader("Ingrese las calificaciones del estudiante")

p1 = st.number_input("Parcial 1", min_value=0.0, max_value=100.0, step=0.1)
p2 = st.number_input("Parcial 2", min_value=0.0, max_value=100.0, step=0.1)
p3 = st.number_input("Parcial 3", min_value=0.0, max_value=100.0, step=0.1)
tp = st.number_input("Trabajos Prácticos (TP)", min_value=0.0, max_value=100.0, step=0.1)
final = st.number_input("Examen Final", min_value=0.0, max_value=100.0, step=0.1)
asistencia = st.selectbox("Porcentaje de Asistencia", list(range(50, 101)))

if st.button("Predecir"):
    bono, tp_modificada, final_modificada = calcular_notas(tp, asistencia, final)

    # Predicción de nota final
    entrada = np.array([[p1, p2, p3, asistencia]])
    nota_predicha = modelo.predict(entrada)[0]
    nota_final = (
        0.1333 * p1 +
        0.1333 * p2 +
        0.1333 * p3 +
        0.20 * tp_modificada +
        0.40 * final_modificada
    )
    clasificacion = clasificar(nota_final)

    # Mostrar resultados
    st.markdown("### Resultados de la Predicción")
    st.write(f"📌 Bono aplicado: {bono:.1f} puntos")
    st.write(f"📌 TP Modificada: {tp_modificada:.1f}")
    st.write(f"📌 Examen Final considerado: {final_modificada:.1f}")
    st.write(f"✅ Nota Final Calculada: {nota_final:.1f}")
    st.write(f"🎓 Clasificación: **{clasificacion}**")
