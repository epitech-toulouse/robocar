import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, FlatList } from 'react-native';
import LogService from '../services/LogService';

const DashboardScreen = () => {
  const [logs, setLogs] = useState<any[]>([]);

  useEffect(() => {
    // Subscribe to logs
    const unsubscribe = LogService.subscribe((newLogs) => {
      setLogs([...newLogs]);
    });
    return unsubscribe;
  }, []);

  const renderLogItem = ({ item }: { item: any }) => {
    const color = item.type === 'error' ? '#ff3b30' : item.type === 'warning' ? '#ff9500' : '#333';
    return (
      <View style={styles.logItem}>
        <Text style={[styles.logTime, { color }]}>[{item.timestamp}]</Text>
        <Text style={[styles.logMessage, { color }]}>{item.message}</Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.logHeader}>
        <Text style={styles.logTitle}>Système Logs</Text>
      </View>
      <FlatList
        data={logs}
        keyExtractor={(item) => item.id}
        renderItem={renderLogItem}
        style={styles.logList}
        contentContainerStyle={{ padding: 10 }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f7',
  },
  logHeader: {
    padding: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e5ea',
  },
  logTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1c1c1e',
  },
  logList: {
    flex: 1,
  },
  logItem: {
    flexDirection: 'row',
    marginBottom: 8,
    backgroundColor: '#fff',
    padding: 10,
    borderRadius: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 1,
    elevation: 1,
  },
  logTime: {
    marginRight: 10,
    fontWeight: '600',
    fontSize: 12,
  },
  logMessage: {
    flex: 1,
    fontSize: 14,
  },
});

export default DashboardScreen;
