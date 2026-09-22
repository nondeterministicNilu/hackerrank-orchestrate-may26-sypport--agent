-- Table: dbo.data

-- DROP TABLE IF EXISTS dbo.data;

CREATE TABLE IF NOT EXISTS dbo.data
(
    content_id text COLLATE pg_catalog."default" NOT NULL,
    company text COLLATE pg_catalog."default",
    file_name text COLLATE pg_catalog."default",
    content text COLLATE pg_catalog."default",
    content_vector vector(512) NOT NULL,
    CONSTRAINT data_pkey PRIMARY KEY (content_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS dbo.data
    OWNER to postgres;