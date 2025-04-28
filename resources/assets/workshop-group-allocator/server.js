// server.js
// ---------
const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs-extra');
const path = require('path');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 3000;
const CONFIG_PATH = process.env.CONFIG_PATH || './config/groups.json';

app.use(bodyParser.json());
app.use(require('cors')());
// Log HTTP requests to stdout (OpenShift captures container stdout)
app.use(morgan('combined'));

// In-memory membership for demo (replace with DB in prod)
let membership = {};

// Load group definitions
let groups = {};
function loadConfig() {
  groups = fs.readJsonSync(CONFIG_PATH);
  console.log('Group config loaded:', Object.keys(groups));
}
loadConfig();
// watch for config changes
fs.watchFile(CONFIG_PATH, () => {
  console.log('Config changed, reloading');
  loadConfig();
});

// Helpers
function logActivity(entry) {
  // Write activity to stdout for OpenShift logging
  console.log(`${new Date().toISOString()} - ${entry}`);
}

// API: get all groups and metadata
app.get('/api/groups', (req, res) => {
  const list = Object.entries(groups).map(([id, meta]) => ({ id, ...meta, count: Object.values(membership).filter(g => g === id).length }));
  res.json(list);
});

// API: get my membership (via ?user=name)
app.get('/api/me', (req, res) => {
  const user = req.query.user;
  if (!user) return res.status(400).json({ error: 'user query required' });
  res.json({ user, group: membership[user] || null });
});

// API: join group
app.post('/api/membership', (req, res) => {
  const { user, group } = req.body;
  if (!user || !group) return res.status(400).json({ error: 'user and group required' });
  if (!groups[group]) return res.status(404).json({ error: 'group not found' });
  membership[user] = group;
  logActivity(`${user} joined ${group}`);
  res.json({ success: true });
});

// API: leave group
app.delete('/api/membership', (req, res) => {
  const { user } = req.body;
  if (!user) return res.status(400).json({ error: 'user required' });
  const prev = membership[user];
  delete membership[user];
  logActivity(`${user} left ${prev}`);
  res.json({ success: true });
});

// Serve UI
app.use('/', express.static(path.join(__dirname, 'public')));

app.listen(PORT, () => console.log(`Service running on port ${PORT}`));
