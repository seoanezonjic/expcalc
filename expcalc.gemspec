# frozen_string_literal: true

require_relative "lib/expcalc/version"

Gem::Specification.new do |spec|
  spec.name          = "expcalc"
  spec.version       = Expcalc::VERSION
  spec.authors       = ["seoanezonjic"]
  spec.email         = ["seoanezonjic@hotmail.com"]

  spec.summary       = "Gem to expand ruby math capabilities"
  spec.description   = "To expand ruby math operations this gem call to others such as Numo:narray and others and implements methods onto them to deal with our needs"
  spec.homepage      = "https://github.com/seoanezonjic/expcalc"
  spec.license       = "MIT"
  spec.required_ruby_version = Gem::Requirement.new(">= 2.4.0")

  #spec.metadata["allowed_push_host"] = "TODO: Set to 'http://mygemserver.com'"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  #spec.metadata["changelog_uri"] = "TODO: Put your gem's CHANGELOG.md URL here."

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{\A(?:test|spec|features)/}) }
  end
  spec.bindir        = "bin"
  spec.executables   = spec.files.grep(%r{\Abin/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "cmath", ">= 1.0.0"
  spec.add_dependency "numo-linalg", ">= 0.1.5"
  spec.add_dependency "numo-narray", ">= 0.9.1.9"
  spec.add_dependency "pp", ">= 0.1.0"
  spec.add_dependency "pycall", ">= 1.3.1"
  spec.add_dependency "npy", ">= 0.2.0"

  spec.add_development_dependency "rake"
  spec.add_development_dependency "rspec"
  # For more information and examples about making a new gem, checkout our
  # guide at: https://bundler.io/guides/creating_gem.html
end
