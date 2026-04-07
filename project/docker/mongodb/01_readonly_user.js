// 01_readonly_user.js
// Creates a read-only MongoDB user for the MCP server / agent layer.
//
// Security principle: principle of least privilege.
// The MCP server query_geodata tool reads from the airbnb_geo database.
// A read-only user ensures the agent layer cannot modify neighbourhood
// boundaries or POI data even if the tool were misused.
//
// Users created:
//   airbnb_admin    — readWrite on airbnb_geo (used by load_mongodb.py)
//   airbnb_readonly — read only on airbnb_geo (used by MCP server)
//
// This script runs automatically on first container start via the
// /docker-entrypoint-initdb.d/ mechanism (mongo:7.0).

db = db.getSiblingDB('airbnb_geo');

// Admin user — full readWrite for transformation scripts
db.createUser({
    user: 'airbnb_admin',
    pwd:  'airbnb_admin_pass',
    roles: [{ role: 'readWrite', db: 'airbnb_geo' }]
});

// Read-only user — for MCP server / agent
db.createUser({
    user: 'airbnb_readonly',
    pwd:  'airbnb_readonly_pass',
    roles: [{ role: 'read', db: 'airbnb_geo' }]
});
