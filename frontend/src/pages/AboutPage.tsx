import { Card, PageHeader, SectionHeading } from '../components/ui'

const STACK = [
  { name: 'React + Vite + TypeScript', role: 'The interface, built as a single-page application with typed API calls.' },
  { name: 'Tailwind CSS', role: 'The design system: layout, colour, type scale and component styling.' },
  { name: 'FastAPI', role: 'The HTTP layer, background training jobs and structured error handling.' },
  { name: 'pandas + NumPy', role: 'Parsing, profiling, filtering and all exploratory analysis.' },
  { name: 'scikit-learn', role: 'Preprocessing pipelines, model training, evaluation and prediction.' },
  { name: 'Plotly.js', role: 'Charts, drawn in the browser from data the backend computes.' },
]

const MILESTONES = [
  { label: 'Milestone 1', body: 'Dataset upload and validation' },
  { label: 'Milestone 2', body: 'Exploratory data analysis dashboard' },
  { label: 'Milestone 3', body: 'Model training and evaluation' },
  { label: 'Milestone 4', body: 'Single and batch prediction' },
  { label: 'Milestone 5', body: 'Migration from Streamlit to React and FastAPI' },
]

const SPECIALISATIONS = [
  'Artificial Intelligence',
  'Machine Learning',
  'Data Science',
  'Python Development',
  'Game Development',
  'Full-stack Web Applications',
]

export function AboutPage() {
  return (
    <div className="space-y-8 max-w-[1100px]">
      <PageHeader
        eyebrow="About"
        title="About EduPredict AI"
        description="An end-to-end machine learning workspace for analysing and predicting student academic performance."
      />

      <Card>
        <p className="text-[15px] text-muted leading-relaxed max-w-[70ch]">
          This application predicts student performance using machine learning. It began as a
          portfolio project and now runs as a React single-page application backed by a FastAPI
          service, with scikit-learn doing the modelling. A dataset goes in, and analysis, a trained
          model and downloadable predictions come out.
        </p>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <SectionHeading title="Mission" />
          <p className="text-[14px] text-muted leading-relaxed">
            Make data-driven insight into student outcomes usable by educators, researchers and
            institutions, without requiring a data science background.
          </p>
        </Card>
        <Card>
          <SectionHeading title="Vision" />
          <p className="text-[14px] text-muted leading-relaxed">
            A world where every learner benefits from early, transparent and actionable performance
            insight, powered by responsible AI.
          </p>
        </Card>
      </div>

      <div>
        <SectionHeading
          title="Technology stack"
          description="The application previously ran on Streamlit. That framework has been removed; nothing here depends on it."
        />
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {STACK.map((item) => (
            <Card key={item.name}>
              <h3 className="font-display text-[14px] font-bold text-ink">{item.name}</h3>
              <p className="text-[13px] text-muted mt-1.5 leading-relaxed">{item.role}</p>
            </Card>
          ))}
        </div>
      </div>

      <Card>
        <SectionHeading
          title="Architecture"
          description="A React client and a Python service, talking over a JSON API."
        />
        <div className="rounded-lg bg-canvas border border-line p-5 font-mono text-[12.5px] text-ink leading-[1.9] overflow-x-auto">
          <div>React (Vite, TypeScript, Tailwind)</div>
          <div className="text-muted">│ fetch /api/… with a session header</div>
          <div>FastAPI</div>
          <div className="text-muted">├── dataset service → parse, profile, page, filter</div>
          <div className="text-muted">├── analysis service → cached chart data per dataset</div>
          <div className="text-muted">├── model service → background training jobs</div>
          <div className="text-muted">└── prediction service → single, batch, CSV export</div>
          <div>scikit-learn pipeline</div>
          <div className="text-muted">
            SimpleImputer → StandardScaler / OneHotEncoder → ColumnTransformer → estimator
          </div>
        </div>
        <p className="text-[13px] text-muted mt-4 leading-relaxed">
          State that used to live in Streamlit's session now lives on the backend behind identifiers:
          a dataset id, a model id, a training job id and a prediction id. Uploading a new dataset
          clears the trained model, and every model records the dataset it was fitted on, so a model
          trained on one file can never be used against another.
        </p>
      </Card>

      <div>
        <SectionHeading title="Timeline" />
        <Card>
          <ol className="space-y-3">
            {MILESTONES.map((milestone) => (
              <li key={milestone.label} className="flex gap-4 text-[14px]">
                <span className="font-semibold text-brand-deep w-28 shrink-0">{milestone.label}</span>
                <span className="text-muted">{milestone.body}</span>
              </li>
            ))}
          </ol>
        </Card>
      </div>

      <div>
        <SectionHeading title="Developer" />
        <Card>
          <p className="font-display text-[20px] font-extrabold text-ink tracking-[-0.02em]">
            Mohamed Shahjan
          </p>
          <p className="text-[14px] text-brand-deep font-semibold mt-0.5">
            AI and Machine Learning Developer
          </p>
          <p className="text-[13px] text-muted mt-0.5">
            Game development enthusiast · B.E. Computer Science Engineering, second year
          </p>

          <div className="flex flex-wrap gap-2 mt-5">
            {SPECIALISATIONS.map((item) => (
              <span
                key={item}
                className="px-2.5 h-7 inline-flex items-center rounded-full bg-canvas border border-line text-[12px] text-ink"
              >
                {item}
              </span>
            ))}
          </div>

          <p className="text-[14px] text-muted mt-5 leading-relaxed max-w-[70ch]">
            A passionate B.E. Computer Science Engineering student with a strong interest in
            artificial intelligence, machine learning, data science, Python development and game
            development. I enjoy building intelligent applications, modern software solutions and
            immersive gaming experiences that solve real-world problems while continuously learning
            emerging technologies. My goal is to become a professional AI engineer and game
            developer.
          </p>
        </Card>
      </div>

      <div>
        <SectionHeading title="Contact" />
        <Card>
          <p className="text-[14px] text-muted">
            Questions or feedback? Reach out through the project repository or your course
            instructor.
          </p>
        </Card>
      </div>
    </div>
  )
}
