from frqi_mcry_representing import FRQI_MCRY_Simulator

if __name__ == "__main__":
    # Define las intensidades de los píxeles (ej. para una imagen 2x2)
    # [píxel_00, píxel_01, píxel_10, píxel_11]
    my_intensities = [20, 60, 120, 240] # Valores de 0 a 255.

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
    my_instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/f1d7b6cfa7094a2c8c9b845ca9d29dad:8c25a8bf-915b-4cc9-86bb-5af77c108e01::"
    # simulator.connect_to_ibm_backend(IBM_QUANTUM_TOKEN)
    #
    # Si quieres un backend específico (ej. 'ibm_lagos'), usa:
    simulator.connect_to_ibm_backend(token=my_api_token)

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