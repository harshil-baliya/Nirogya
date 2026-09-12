-- Sync existing production DB state with current Prisma schema.
-- This migration is idempotent so it can safely run across environments.

create extension if not exists "pgcrypto";

create table if not exists public.profiles (
  id uuid primary key default gen_random_uuid(),
  uid uuid unique not null,
  name text,
  email text,
  language text,
  city text,
  age_group text,
  life_stage text,
  activity_level text,
  diet_type text,
  conditions text[] not null default '{}'::text[],
  family_history text[] not null default '{}'::text[],
  medications text,
  allergies text[] not null default '{}'::text[],
  sleep_hours numeric(4,1),
  sleep_quality text[] not null default '{}'::text[],
  stress_level text,
  mood_swings boolean,
  anxiety boolean,
  low_motivation boolean,
  last_active_at timestamptz,
  deleted_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.assessments
  add column if not exists deleted_at timestamptz;

create index if not exists idx_assessments_uid_created_at
  on public.assessments (uid, created_at desc);

create index if not exists idx_profiles_uid
  on public.profiles (uid);

insert into public.profiles (
  uid,
  created_at,
  updated_at,
  last_active_at
)
select distinct
  a.uid,
  now(),
  now(),
  now()
from public.assessments a
left join public.profiles p on p.uid = a.uid
where p.uid is null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'assessments_uid_fkey'
      and conrelid = 'public.assessments'::regclass
  ) then
    alter table public.assessments
      add constraint assessments_uid_fkey
      foreign key (uid)
      references public.profiles(uid)
      on delete cascade;
  end if;
end
$$;

alter table public.profiles enable row level security;
alter table public.assessments enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
drop policy if exists "profiles_upsert_own" on public.profiles;
drop policy if exists "assessments_select_own" on public.assessments;
drop policy if exists "assessments_insert_own" on public.assessments;
drop policy if exists "assessments_update_own" on public.assessments;

grant usage on schema public to anon, authenticated;
grant select, insert, update on public.profiles to anon, authenticated;
grant select, insert, update on public.assessments to anon, authenticated;

create policy "profiles_select_own" on public.profiles
  for select using (uid = auth.uid() and deleted_at is null);

create policy "profiles_upsert_own" on public.profiles
  for all
  using (uid = auth.uid())
  with check (uid = auth.uid());

create policy "assessments_select_own" on public.assessments
  for select using (uid = auth.uid() and deleted_at is null);

create policy "assessments_insert_own" on public.assessments
  for insert with check (uid = auth.uid());

create policy "assessments_update_own" on public.assessments
  for update using (uid = auth.uid()) with check (uid = auth.uid());
