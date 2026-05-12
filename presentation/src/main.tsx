import React from 'react';
import ReactDOM from 'react-dom/client';
import { Presentation } from './presentation/Presentation';
import './presentation/presentation.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Presentation />
  </React.StrictMode>
);
