-- Genetic profile & risk flags tables (were missing from migration history)

create table if not exists public.genetic_profiles (
  id uuid primary key default gen_random_uuid(),
  uid uuid unique not null,
  rs_ids text[] not null default '{}'::text[],
  prs_t2d numeric(6,4) not null,
  prs_cad numeric(6,4) not null,
  prs_htn numeric(6,4) not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint genetic_profiles_uid_fkey
    foreign key (uid) references public.profiles(uid) on delete cascade
);

create index if not exists idx_genetic_profiles_uid
  on public.genetic_profiles (uid);

create table if not exists public.genetic_flags (
  id uuid primary key default gen_random_uuid(),
  uid uuid not null,
  genetic_profile_id uuid not null,
  type text not null,
  gene text,
  severity text not null,
  conditions text[] not null default '{}'::text[],
  plain_language text not null,
  action_required text not null,
  drug_warning text,
  source text not null,
  created_at timestamptz not null default now(),
  constraint genetic_flags_uid_fkey
    foreign key (uid) references public.profiles(uid) on delete cascade,
  constraint genetic_flags_genetic_profile_id_fkey
    foreign key (genetic_profile_id) references public.genetic_profiles(id) on delete cascade
);

create index if not exists idx_genetic_flags_uid_severity
  on public.genetic_flags (uid, severity);

create index if not exists idx_genetic_flags_profile_id
  on public.genetic_flags (genetic_profile_id);
