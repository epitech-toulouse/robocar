type Log = {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'error' | 'warning';
};

class LogService {
  private logs: Log[] = [];
  private listeners: ((logs: Log[]) => void)[] = [];

  constructor() {
    // Initial dummy logs
    this.addLog('Initialisation du système', 'info');
    this.addLog('En attente de connexion Bluetooth', 'warning');
  }

  getLogs() {
    return this.logs;
  }

  addLog(message: string, type: 'info' | 'error' | 'warning' = 'info') {
    const newLog: Log = {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toLocaleTimeString(),
      message,
      type,
    };
    
    // Add to the beginning of the array so latest is first
    this.logs = [newLog, ...this.logs];
    
    // Notify listeners
    this.listeners.forEach(listener => listener(this.logs));
  }

  clearLogs() {
    this.logs = [];
    this.listeners.forEach(listener => listener(this.logs));
  }

  subscribe(listener: (logs: Log[]) => void) {
    this.listeners.push(listener);
    listener(this.logs); // Immediately send current logs
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }
}

export default new LogService();
