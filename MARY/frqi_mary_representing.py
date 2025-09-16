# Importaciones de librerías comunes
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

class FRQI_MARY_Simulator:
    """
    Simulador del algoritmo FRQI usando la implementación MARY del artículo,
    con los métodos de pre y postprocesamiento corregidos según el estudio.
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
        ## CORRECCIÓN ##
        Implementa la transformación lineal para convertir intensidades en ángulos,
        según la Ecuación (8) del artículo.
        """
        # Normaliza intensidades al rango [0, 1] y aplica la transformación
        thetas = [(i / self.max_intensity) * (np.pi / 2) for i in intensities]
        return thetas

    def connect_to_ibm_backend(self, token: str, backend_name: str = None, channel: str = "ibm_quantum_platform"):
        """
        Conecta al servicio IBM Quantum y guarda el objeto backend.
        """
        try:
            self.service = QiskitRuntimeService(channel=channel, token=token)
            print(f"Conexión establecida con IBM Quantum Platform (canal: {channel}).")

            if backend_name:
                self.backend = self.service.backend(backend_name)
            else:
                backends = self.service.backends(simulator=False, operational=True)
                real_backends = [b for b in backends if b.num_qubits >= 3]
                if real_backends:
                    self.backend = sorted(real_backends, key=lambda b: b.status().pending_jobs)[0]
                else:
                    raise RuntimeError("No hay backends cuánticos reales disponibles con ≥ 3 qubits.")
            
            print(f"Backend seleccionado: {self.backend.name} (Trabajos pendientes: {self.backend.status().pending_jobs})")

        except IBMRuntimeError as e:
            print(f"Error al conectar con IBM Quantum Platform: {e}")
            self.service = None
            self.backend = None
        except Exception as e:
            print(f"Ocurrió un error inesperado durante la conexión: {e}")
            self.service = None
            self.backend = None

    def mary_gate(self, qc, angle, control_qubits, target_qubit):
        """Implementación de la CCRY-Gate (MARY) del artículo."""
      
        #Aplicando la primera compuerta Ry
        qc.ry(angle / 2, target_qubit)
      
        #Aplicando el primer CNOT
        qc.cx(control_qubits[0], target_qubit)

        #Aplicando la segunda compuerta Ry
        qc.ry(-angle / 2, target_qubit)

        #Aplicando el segundo CNOT
        qc.cx(control_qubits[1], target_qubit)

        #Aplicando la tercera compuerta Ry
        qc.ry(angle / 2, target_qubit)

        #Aplicando la tercera CNOT
        qc.cx(control_qubits[0], target_qubit)

        #Aplicando la cuarta compuerta Ry
        qc.ry(-angle / 2, target_qubit)

        #Aplicando el segundo CNOT
        qc.cx(control_qubits[1], target_qubit)

    def build_circuit(self):
        """
        ## CORRECCIÓN ##: Se eliminó 'self.' al llamar a las clases de Qiskit.
        """
        qr = QuantumRegister(3, name='q')
        cr = ClassicalRegister(3, name='c')
        qc = QuantumCircuit(qr, cr, name="FRQI_MARY")

        # Aplicar Hadamard a los qubits de posición para crear superposición
        qc.h(qr[1])  # pos1 (q[1] es el MSB de la posición)
        qc.h(qr[2])  # pos0 (q[2] es el LSB de la posición)

        control_qubits = [qr[1], qr[2]]
        target_qubit = qr[0]

        for i,theta in enumerate(self.thetas):

            self.mary_gate(
                qc = qc, 
                angle = 2*theta, 
                control_qubits = control_qubits,
                target_qubit = target_qubit
            )

            # Apply X gates based on the image's pattern.
            if i == 0:
                qc.x(qr[1])
            elif i == 1:
                qc.x(qr[1])
                qc.x(qr[2])
            elif i == 2:
                qc.x(qr[1])

        # Medir todos los qubits y almacenar los resultados en los registros clásicos
        qc.barrier(qr)
        qc.measure(qr, cr)
        self.qc = qc

    def run(self):
        """Ejecuta el circuito en el backend configurado."""
        self.build_circuit()
        self.counts = None

        if self.backend and self.service:
            print(f"\nEjecutando en el backend de IBM: {self.backend.name} con {self.shots} shots...")
            try:
                sampler = Sampler(mode=self.backend)
                transpiled_qc = transpile(self.qc, self.backend)
                
                job = sampler.run([transpiled_qc], shots=self.shots)
                print(f"Job ID: {job.job_id()} enviado a {self.backend.name}.")
                print("Esperando resultados...")
                
                result = job.result()
                
                pub_result = result[0]
                self.counts = pub_result.data.c.get_counts()

                print("Resultados obtenidos del backend de IBM:")
                print(self.counts)

            except Exception as e:
                print(f"Error al ejecutar en el backend de IBM: {e}")
                print("Volviendo al simulador local.")
                self._run_local_simulator()
        else:
            print("\nBackend de IBM no disponible. Ejecutando en el simulador QASM local...")
            self._run_local_simulator()

    def _run_local_simulator(self):
        """Ejecuta el circuito en el simulador local."""
        try:
            simulator = Aer.get_backend('qasm_simulator')
            transpiled_qc = transpile(self.qc, simulator)
            job = simulator.run(transpiled_qc, shots=self.shots)
            result = job.result()
            self.counts = result.get_counts(transpiled_qc) 
            print("Resultados obtenidos del simulador QASM local:")
            print(self.counts)
        except Exception as e:
            print(f"Error al ejecutar en el simulador local: {e}")
            self.counts = {}

    def analyze_results(self):
        """
        ## CORRECCIÓN ##
        Analiza los resultados usando la Ecuación (10) del artículo y una
        lógica de bits estandarizada.
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        print(f"\nCantidad de simulaciones: {self.shots}")
        print("\nMediciones del qubit de color q[0] para cada posición:")
        self.results = {}
        self.plot_probabilities_data = {} # Reinicializar aquí

        positions_order = ['00', '01', '10', '11'] # Orden de las intensidades originales

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
        reconstructed_pixels[0, 0] = self.results['11']['reconstructed_intensity']
        reconstructed_pixels[0, 1] = self.results['00']['reconstructed_intensity']
        reconstructed_pixels[1, 0] = self.results['10']['reconstructed_intensity']
        reconstructed_pixels[1, 1] = self.results['01']['reconstructed_intensity']

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

    def draw_circuit(self, output='mpl', filename='quantum_circuit.png', style=None):
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