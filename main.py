from frqi_mcry_representing import FRQI_MCRY_Simulator
from frqi_mary_representing import FRQI_MARY_Simulator

if __name__ == "__main__":
    # Define las intensidades de los píxeles (ej. para una imagen 2x2)
    # [píxel_00, píxel_01, píxel_10, píxel_11]
    my_intensities = [20, 60, 120, 240] # Valores de 0 a 255.

    #MCRY
    ########################################################
    # Crea una instancia del simulador
    n = 150000
    simulator = FRQI_MCRY_Simulator(intensities=my_intensities, shots=n) # Reducir shots para pruebas rápidas

    # --- Opcional: Conectar a una computadora cuántica real de IBM ---
    # Necesitarás tu token de API de IBM Quantum Experience.
    # Puedes encontrarlo en https://quantum.ibm.com/account
    # Descomenta las siguientes líneas y reemplaza "YOUR_IBM_QUANTUM_TOKEN_HERE"
    # con tu token real si quieres usar un backend real.
    #
    my_api_token = "amHoPNxpidFKy89LRK8bgyHUkOsVsaMup-dLc9fZdcQ7"
    my_instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/f1d7b6cfa7094a2c8c9b845ca9d29dad:d1467f7d-3360-4d28-8fef-9025fea31956::"
    # simulator.connect_to_ibm_backend(IBM_QUANTUM_TOKEN)
    #
    # Si quieres un backend específico (ej. 'ibm_lagos'), usa:
    simulator.connect_to_ibm_backend(
        token=my_api_token,
        instance=my_instance,
        backend_name=None  # o déjalo como None para seleccionar automáticamente
    )

    print("EJECUTANDO CIRCUITO CUÁNTICO CON COMPUERTAS MCRY")
    # Ejecuta el circuito (usará el backend de IBM si se conectó, de lo contrario, el simulador local)
    simulator.run()

    # Analiza los resultados
    simulator.analyze_results()

    # Muestra el histograma de los conteos de medición
    simulator.plot_histogram()

    # Muestra el gráfico de probabilidades
    simulator.plot_probabilities()

    # Muestra la imagen reconstruida (nueva función)
    simulator.reconstruct_image()

    # Muestra la tabla de estados esperados
    simulator.print_state_table()

    #Mary
    ########################################################
    print("EJECUTANDO CIRCUITO CUÁNTICO CON COMPUERTAS MCRY")
    simulator_mary = FRQI_MCRY_Simulator(intensities=my_intensities, shots=n) # Reducir shots para pruebas rápidas
    simulator_mary.connect_to_ibm_backend(
        token=my_api_token,
        instance=my_instance,
        backend_name=None  # o déjalo como None para seleccionar automáticamente
    )

    # Ejecuta el circuito (usará el backend de IBM si se conectó, de lo contrario, el simulador local)
    #simulator_mary.run()

    # Analiza los resultados
    #simulator_mary.analyze_results()

    # Muestra el histograma de los conteos de medición
    #simulator_mary.plot_histogram()

    # Muestra el gráfico de probabilidades
    #simulator_mary.plot_probabilities()

    # Muestra la imagen reconstruida (nueva función)
    #simulator_mary.reconstruct_image()

    # Muestra la tabla de estados esperados
    #simulator_mary.print_state_table()

