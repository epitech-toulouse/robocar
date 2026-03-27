import React from 'react';
import { createMaterialTopTabNavigator } from '@react-navigation/material-top-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import DashboardScreen from '../screens/DashboardScreen';
import ControlScreen from '../screens/ControlScreen';
import ConnectionScreen from '../screens/ConnectionScreen';
import MapScreen from '../screens/MapScreen';

const Tab = createMaterialTopTabNavigator();
const Stack = createNativeStackNavigator();

const MainTabs = () => {
  return (
    <Tab.Navigator
      initialRouteName="Dashboard"
      screenOptions={{
        swipeEnabled: false,
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: 'gray',
        tabBarIndicatorStyle: { backgroundColor: '#007AFF' },
        tabBarStyle: { backgroundColor: '#fff' },
      }}
    >
      <Tab.Screen 
        name="Dashboard" 
        component={DashboardScreen} 
        options={{ tabBarLabel: 'Logs' }} 
      />
      <Tab.Screen 
        name="Control" 
        component={ControlScreen} 
        options={{ tabBarLabel: 'Contrôles' }} 
      />
      <Tab.Screen 
        name="Navigation" 
        component={MapScreen} 
        options={{ tabBarLabel: 'Navigation' }} 
      />
    </Tab.Navigator>
  );
};

const AppNavigator = () => {
  return (
    <Stack.Navigator 
      initialRouteName="Connection"
      screenOptions={{ headerShown: false }}
    >
      <Stack.Screen name="Connection" component={ConnectionScreen} />
      <Stack.Screen name="Main" component={MainTabs} />
    </Stack.Navigator>
  );
};

export default AppNavigator;
