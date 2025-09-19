import os
import dotenv
from frqi_mary_corr import FRQI_MARY_CORR_Simulator

if __name__ == "__main__":
    # Define las intensidades de los píxeles
    my_intensities = [20, 60, 120, 240]

    # Crea una instancia del simulador
    n_shots = 8192 #8192
    simulator = FRQI_MARY_CORR_Simulator(intensities=my_intensities, shots=n_shots)

    # cargamos las variables de entorno
    dotenv.load_dotenv()  

    # --- Conexión a una computadora cuántica real de IBM (Actualizado 2025) ---
    my_api_token = os.getenv('my_api_token') # Reemplaza con tu token real de IBM Cloud
    
    # Opcional: Proporciona tu instancia (CRN) para optimizar la búsqueda de backends.
    # Si no la pones, Qiskit la buscará por ti.
    my_instance = os.getenv('my_instance')

    # Conectar al backend de IBM usando el nuevo canal y las credenciales actualizadas
    #simulator.connect_to_ibm_backend(
    #    token=my_api_token,
    #    backend_name="ibm_torino", # O el nombre del backend que prefieras, o None
        # El canal 'ibm_quantum' está obsoleto. Se usa 'ibm_quantum_platform'.
        #Para correr en el simulador poner por favor 
    #    channel="ibm_quantum_platform"
    #)

    print("\nEJECUTANDO CIRCUITO CUÁNTICO CON COMPUERTAS MARY")
    simulator.run()

    # El resto del flujo de trabajo permanece igual
    simulator.draw_circuit(output='mpl')
    simulator.analyze_results()
    simulator.plot_histogram()
    simulator.plot_probabilities()
    simulator.reconstruct_image()
    simulator.print_state_table()