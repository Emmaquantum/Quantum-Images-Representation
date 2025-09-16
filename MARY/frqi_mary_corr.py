# Importaciones de librerías comunes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# Importaciones de Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import RYGate
from qiskit.visualization import plot_histogram as qiskit_plot_histogram
from qiskit.visualization import circuit_drawer

# Importaciones para IBM Quantum (nueva plataforma 2025+)
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit_ibm_runtime.exceptions import IBMRuntimeError


#Apagar algo
from importlib.metadata import version
version("qiskit")



class FRQI_MARY_CORR_Simulator:
    """
    Simulador del algoritmo FRQI (Flexible Representation of Quantum Images) con corrección de errores de 3 qubits.

    Esta clase permite codificar una imagen (representada por intensidades de píxeles)
    en un circuito cuántico utilizando la codificación FRQI, ejecutar el circuito
    en un simulador local o en una computadora cuántica real de IBM, y analizar
    los resultados.
    """

    def __init__(self, intensities=None, max_intensity=255, shots=8000, n=2):
        """
        Inicializa el simulador FRQI.

        Args:
            intensities (list, optional): Lista de intensidades de píxeles (0-255).
                                         Por defecto, [15, 25, 150, 255].
            max_intensity (int, optional): Valor máximo de intensidad para normalización.
                                           Por defecto, 255.
            shots (int, optional): Número de veces que se ejecuta el circuito cuántico.
                                   Por defecto, 80000.
        """
        # Atributos para IBM Quantum
        self.service = None # Instancia de QiskitRuntimeService
        self.backend = None # El backend a usar (simulador o real) - ahora almacenará el NOMBRE del backend
                            # ya que Sampler toma el nombre, no un objeto backend
        self.backend_object = None # This will store the actual backend object if retrieved for properties/status

        # Variables para correr el código
        self.n = n
        self.max_intensity = max_intensity
        self.shots = shots
        self.intensities = intensities if intensities is not None else [15, 25, 150, 255]

        # Calcular thetas desde intensidades
        self.thetas = self.calculate_thetas(self.intensities)

        self.results = {}
        self.probabilities_data = {}
        self.counts = None
        self.qc = None

    def calculate_thetas(self, intensities):
        """
        Calcula los ángulos theta para la codificación FRQI a partir de las intensidades.

        Args:
            intensities (list): Lista de intensidades de píxeles.

        Returns:
            list: Lista de ángulos theta en radianes.
        """
        # Normaliza intensidades al rango [0, 1]
        intensity_norm = [i / self.max_intensity for i in intensities]
        #intensity_norm = [min(1.0, max(-1.0, i)) for i in intensity_norm]
        # Calcula thetas con la fórmula FRQI: theta = 2 * arcsin(intensidad_normalizada)
        #thetas = [np.arcsin(i) for i in intensity_norm]
        thetas = [(np.pi/2)*i for i in intensity_norm]
        return thetas

    def connect_to_ibm_backend(self, token: str, backend_name: str = None, channel: str = "ibm_quantum_platform"):
        """
        ## ACTUALIZACIÓN JULIO 2025 ##
        Conecta al servicio IBM Quantum.
        El 'channel' por defecto es ahora 'ibm_quantum_platform', que unifica
        el acceso gratuito (Open Plan) y de pago. El antiguo 'ibm_quantum' está obsoleto.
        """
        try:
            self.service = QiskitRuntimeService(channel=channel, token=token)
            print(f" Conexión establecida con IBM Quantum Platform (canal: {channel}).")

            if backend_name:
                self.backend = self.service.backend(backend_name)
            else:
                # Lógica para encontrar el mejor backend disponible
                backends = self.service.backends(simulator=False, operational=True)
                real_backends = [b for b in backends if b.num_qubits >= 3]
                if real_backends:
                    self.backend = sorted(real_backends, key=lambda b: b.status().pending_jobs)[0]
                else:
                    raise RuntimeError("No hay backends cuánticos reales disponibles con ≥ 3 qubits.")
            
            print(f"Backend seleccionado: {self.backend.name} (Trabajos pendientes: {self.backend.status().pending_jobs})")

        except IBMRuntimeError as e:
            print(f" Error al conectar con IBM Quantum Platform: {e}")
            self.service = None
            self.backend = None
        except Exception as e:
            print(f" Ocurrió un error inesperado durante la conexión: {e}")
            self.service = None
            self.backend = None

    def mary_gate(self, qc, angle, control_qubits, target_qubit):
        """Implementación de la CCRY-Gate (MARY) del artículo."""
      
        #Aplicando la primera compuerta Ry
        qc.ry(angle / 4, target_qubit)
      
        #Aplicando el primer CNOT
        qc.cx(control_qubits[0], target_qubit)

        #Aplicando la segunda compuerta Ry
        qc.ry(-angle / 4, target_qubit)

        #Aplicando el segundo CNOT
        qc.cx(control_qubits[1], target_qubit)

        #Aplicando la tercera compuerta Ry
        qc.ry(angle / 4, target_qubit)

        #Aplicando la tercera CNOT
        qc.cx(control_qubits[0], target_qubit)

        #Aplicando la cuarta compuerta Ry
        qc.ry(-angle / 4, target_qubit)

        #Aplicando el segundo CNOT
        qc.cx(control_qubits[1], target_qubit)

        
    def build_circuit(self):
        """
        Construye el circuito cuántico FRQI con corrección de errores de 3 qubits.

        El circuito ahora usa 5 qubits:
        - q[0], q[3], q[4]: qubits de color codificados (q[0] es el lógico)
        - q[1], q[2]: qubits de posición (para 4 píxeles, 2x2 imagen)
        """
        # Crear registros cuánticos y clásicos
        qr = QuantumRegister(5, name='q')  # 5 qubits
        cr = ClassicalRegister(3, name='c')  # 3 bits clásicos para la medición final
        
        # 2 bits clásicos adicionales para el síndrome (q[3] y q[4])
        cr_syndrome = ClassicalRegister(2, name='syndrome') 
        qc = QuantumCircuit(qr, cr_syndrome, cr, name="FRQI_MCRY_QEC")

        # --- Parte 1: Codificación de Estado FRQI ---
        
        # Aplicar Hadamard a los qubits de posición para crear superposición
        qc.h(qr[1])  # pos1 (q[1] es el MSB de la posición)
        qc.h(qr[2])  # pos0 (q[2] es el LSB de la posición)

        # Codificación del qubit de color q[0] en 3 qubits físicos (q[0], q[3], q[4])
        qc.cx(qr[0], qr[3])
        qc.cx(qr[0], qr[4])

        # --- Parte 2: Aplicación de la compuerta FRQI con MC-RY ---
        # Esta compuerta debe aplicarse a la representación lógica del estado
        
        # Aquí simplificamos aplicando la compuerta a los 3 qubits físicos.
        # En una implementación real, se usarían compuertas tolerantes a fallos.
        control_qubits = [qr[1], qr[2]]
        
        for i, theta in enumerate(self.thetas):
            self.mary_gate(
                qc=qc,
                angle=2*theta,
                control_qubits=control_qubits,
                target_qubit=qr[0]
            )
            # Replicar la operación en los qubits de codificación
            self.mary_gate(
                qc=qc,
                angle=2*theta,
                control_qubits=control_qubits,
                target_qubit=qr[3]
            )
            self.mary_gate(
                qc=qc,
                angle=2*theta,
                control_qubits=control_qubits,
                target_qubit=qr[4]
            )
            
            # Aplicar X gates basados en el patrón de la imagen.
            if i == 0:
                qc.x(qr[1])
            elif i == 1:
                qc.x(qr[1])
                qc.x(qr[2])
            elif i == 2:
                qc.x(qr[1])

        # --- Parte 3: Detección y corrección de errores (Decodificación) ---
        qc.barrier()
        
        # Síndrome de medición
        qc.cx(qr[0], qr[3])
        qc.cx(qr[0], qr[4])
        qc.measure([qr[3], qr[4]], [cr_syndrome[0], cr_syndrome[1]])

        # Corrección de errores
        with qc.if_test((cr_syndrome, 1)):
            qc.x(qr[0])
        with qc.if_test((cr_syndrome, 2)):
            qc.x(qr[3])
        with qc.if_test((cr_syndrome, 3)):
            qc.x(qr[4])

        # Medición final de los qubits
        qc.barrier(qr)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])

        self.qc = qc 

    def run(self):
        self.build_circuit()
        self.counts = None

        if self.backend and self.service:
            print(f"\nEjecutando en el backend de IBM: {self.backend.name} con {self.shots} shots...")
            try:
                # Se inicializa el Sampler especificando el 'mode' de ejecución.
                sampler = Sampler(mode=self.backend)
                
                transpiled_qc = transpile(self.qc, self.backend)

                # ## CORRECCIÓN FINAL ##
                # El método run espera una LISTA de circuitos.
                job = sampler.run([transpiled_qc], shots=self.shots)
                print(f"Job ID: {job.job_id()} enviado a {self.backend.name}.")
                print("Esperando resultados...")
                
                result = job.result()
                
                # Como enviamos una lista con un circuito, obtenemos el primer resultado.
                pub_result = result[0]
                self.counts = pub_result.data.c.get_counts()

                print(" Resultados obtenidos del backend de IBM:")
                print(self.counts)

            except Exception as e:
                print(f" Error al ejecutar en el backend de IBM: {e}")
                print(" Volviendo al simulador local.")
                self._run_local_simulator()
        else:
            print("\nBackend de IBM no disponible. Ejecutando en el simulador QASM local...")
            self._run_local_simulator()

    def _run_local_simulator(self):
        """
        Ejecuta el circuito cuántico en el simulador QASM local de Qiskit Aer.
        """
        try:
            simulator = Aer.get_backend('aer_simulator') # <-- AQUI ESTA EL CAMBIO
            transpiled_qc = transpile(self.qc, simulator)
            job = simulator.run(transpiled_qc, shots=self.shots)
            result = job.result()
            self.counts = result.get_counts(transpiled_qc) 
            print(" Resultados obtenidos del simulador QASM local:")
            print(self.counts)
        except Exception as e:
            print(f" Error al ejecutar en el simulador local: {e}")
            self.counts = {} # Asegura que self.counts sea un diccionario vacío si el simulador falla

    def analyze_results(self):
        """
        Analiza los resultados de las mediciones del circuito para extraer
        las probabilidades de color para cada posición y reconstruir la intensidad.
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        print(f"\nCantidad de simulaciones: {self.shots}")
        print("\nMediciones del qubit de color q[0] para cada posición:")
        self.results = {}
        self.plot_probabilities_data = {} # Reinicializar aquí

        positions_order = ['00', '01', '10', '11'] # Orden de las intensidades originales
        prob = []

        for i, target_pos_str in enumerate(positions_order):

            filtered_counts = {'0': 0, '1': 0} # Inicializar conteos para '0' y '1' del qubit de color
            #print(filtered_counts)

            for bitstring, count in self.counts.items():
                color_bit = bitstring[2] # q[0]
                pos1_bit = bitstring[1] # q[1]
                pos0_bit = bitstring[0] # q[2]

                current_pos_str = pos1_bit + pos0_bit

                if current_pos_str == target_pos_str:
                    filtered_counts[color_bit] += count

            total = sum(filtered_counts.values()) # Conteo total para esta posición
            print(f"\nPosición |{target_pos_str}> (de q[1]q[2] medido):")
            if total > 0:
                p0 = filtered_counts.get('0', 0) / total
                p1 = filtered_counts.get('1', 0) / total
            else:
                p0, p1 = 0.0, 0.0
            
            if p0 > 0:
                reconstructed_intensity = np.sqrt(p0)
                reconstructed_intensity = np.arccos(reconstructed_intensity) * (self.max_intensity * (2/np.pi))
                #reconstructed_intensity = np.sqrt(p1)*255
            else:
                reconstructed_intensity = 0.0

            self.results[target_pos_str] = {
                '0': p0,
                '1': p1,
                'reconstructed_intensity': reconstructed_intensity
            }

            self.plot_probabilities_data[f'|{target_pos_str}> (P(0))'] = p0
            self.plot_probabilities_data[f'|{target_pos_str}> (P(1))'] = p1

            print(f"   Posición |{target_pos_str}⟩ (de q[1]q[2] medido):")
            print(f"     Probabilidad de 0 (medido en q[2]): {p0:.4f}")
            print(f"     Probabilidad de 1 (medido en q[1]): {p1:.4f}")

    def plot_histogram(self):
        """
        Muestra un histograma de los conteos de medición del circuito.
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        counts_str_keys = {str(k): v for k, v in self.counts.items()}

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(counts_str_keys.keys()), y=list(counts_str_keys.values()))

        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', label_type='edge')

        plt.title('Valores de Medición por base computacional')
        plt.ylabel('Cuentas')
        plt.xlabel('Estado Medido (q[2]q[1]q[0])')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_probabilities(self):
        """
        Genera un gráfico de barras mostrando las probabilidades de cada estado
        medido (q[2]q[1]q[0]).
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        probabilities = {k: v / self.shots for k, v in self.counts.items()}
        probabilities_str_keys = {str(k): v for k, v in probabilities.items()}

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(probabilities_str_keys.keys()), y=list(probabilities_str_keys.values()))

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', label_type='edge')

        plt.title('Probabilidades de Medición por base computacional')
        plt.ylabel('Probabilidad')
        plt.xlabel('Estado Medido (q[2]q[1]q[0])')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def reconstruct_image(self):
        if not self.results:
            raise RuntimeError("Primero debes ejecutar analyze_results() para obtener resultados.")

        print("\n--- Reconstrucción de la Imagen ---")
        reconstructed_pixels = np.zeros((2, 2), dtype=int)
        reconstructed_pixels[0, 0] = self.results['00']['reconstructed_intensity']
        reconstructed_pixels[0, 1] = self.results['10']['reconstructed_intensity']
        reconstructed_pixels[1, 0] = self.results['01']['reconstructed_intensity']
        reconstructed_pixels[1, 1] = self.results['11']['reconstructed_intensity']

        print("\nIntensidades Originales (Input):")
        original_pixels_display = np.array([
            [self.intensities[0], self.intensities[1]],
            [self.intensities[2], self.intensities[3]]
        ])

        print("\nIntensidades Reconstruidas (Output):")
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original_pixels_display, cmap='gray', vmin=0, vmax=self.max_intensity)
        axes[0].set_title('Imagen Original (Input)')
        axes[0].set_xticks(np.arange(2))
        axes[0].set_yticks(np.arange(2))
        axes[0].set_xticklabels(['Col 0', 'Col 1'])
        axes[0].set_yticklabels(['Fila 0', 'Fila 1'])
        for (j, i), val in np.ndenumerate(original_pixels_display):
            axes[0].text(i, j, f'{val}', ha='center', va='center', color='red', fontsize=12)

        axes[1].imshow(reconstructed_pixels, cmap='gray', vmin=0, vmax=self.max_intensity)
        axes[1].set_title('Imagen Reconstruida (Output)')
        axes[1].set_xticks(np.arange(2))
        axes[1].set_yticks(np.arange(2))
        axes[1].set_xticklabels(['Col 0', 'Col 1'])
        axes[1].set_yticklabels(['Fila 0', 'Fila 1'])
        for (j, i), val in np.ndenumerate(reconstructed_pixels):
            axes[1].text(i, j, f'{val}', ha='center', va='center', color='red', fontsize=12)

        plt.tight_layout()
        plt.show()


    def print_state_table(self):
        """
        Muestra una tabla con las intensidades normalizadas, ángulos theta,
        y los estados parciales esperados para cada posición.
        """
        positions = ['00', '01', '10', '11']
        cos_vals = [round(np.cos(theta), 5) for theta in self.thetas]
        sin_vals = [round(np.sin(theta), 5) for theta in self.thetas]

        data = {
            'Posición $|i>$': [f'$|{p}>$' for p in positions],
            'Intensidad Normalizada': [f'{i:.0f}' for i in self.intensities],
            f'theta_i (rad)': [f'{theta:.5f}' for theta in self.thetas],
            'cos(theta_i)': [f'{c:.5f}' for c in cos_vals],
            'sin(theta_i)': [f'{s:.5f}' for s in sin_vals],
            'Estado Parcial Esperado': [
                f"$({cos_vals[i]}|0> + {sin_vals[i]}|1>)|{positions[i]}>$"
                for i in range(4)
            ]
        }

        df = pd.DataFrame(data)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nTabla de Estados Esperados (Ideal):")
        print(df.to_string(index=False))

    def draw_circuit(self, output='mpl', filename='quantum_circuit_corr.png', style=None):
        """
        Dibuja el circuito cuántico y lo guarda en un archivo.

        Args:
            output (str, optional): Formato de salida del dibujo.
                                    Por defecto, 'mpl'.
            filename (str, optional): Nombre del archivo para guardar la imagen.
                                    Por defecto, 'quantum_circuit.png'.
            style (dict, optional): Un diccionario de estilos para el dibujante.
        """
        if self.qc is None:
            self.build_circuit()

        print(f"\n--- Dibujando el Circuito Cuántico ({output}) ---")

        if output == 'mpl':
            # circuit_drawer returns a Matplotlib figure object
            fig = circuit_drawer(self.qc, output=output, idle_wires=False, style="clifford")
            
            # Guarda la figura en el archivo
            fig.savefig(filename, bbox_inches='tight')
            print(f"Circuito guardado como '{filename}'")
            plt.close(fig)

        elif output == 'text':
            # Imprime el circuito en la terminal
            print(circuit_drawer(self.qc, output=output, idle_wires=False))

        else:
            try:
                # Para otros formatos como 'latex' o 'latex_source'
                result = circuit_drawer(self.qc, output=output, filename=filename, idle_wires=False, style=style)
                
                if filename:
                    print(f"Circuito guardado como '{filename}'")
                elif isinstance(result, str):
                    print(result)

            except Exception as e:
                print(f"No se pudo dibujar el circuito con el formato '{output}'. Error: {e}")
                print("Asegúrate de tener las dependencias necesarias instaladas.")