-- 01_readonly_role.sql
-- Creates a read-only PostgreSQL role for the MCP server / agent layer.
--
-- Security principle: principle of least privilege (Saltzer & Schroeder, 1975).
-- The MCP server exposes SQL execution to an LLM agent. Even though the tool
-- restricts queries to SELECT statements in code, a database-level read-only
-- role provides a second, independent enforcement layer — defence in depth.
--
-- Role: airbnb_readonly
--   - Can connect to the airbnb database
--   - Can SELECT on all tables in the public schema
--   - Cannot INSERT, UPDATE, DELETE, DROP, or CREATE
--
-- This script runs automatically on first container start via the
-- /docker-entrypoint-initdb.d/ mechanism (postgres:15-alpine).

\connect airbnb

-- Create role if it doesn't already exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airbnb_readonly') THEN
        CREATE ROLE airbnb_readonly WITH LOGIN PASSWORD 'airbnb_readonly_pass';
    END IF;
END
$$;

-- Grant connection privilege
GRANT CONNECT ON DATABASE airbnb TO airbnb_readonly;

-- Grant SELECT on all existing tables
GRANT USAGE ON SCHEMA public TO airbnb_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO airbnb_readonly;

-- Grant SELECT on all future tables (applied when transformation scripts load new data)
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO airbnb_readonly;
